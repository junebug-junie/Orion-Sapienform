# orion/core/bus/async_service.py
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from pydantic import BaseModel, ValidationError
from redis import asyncio as aioredis

from orion.schemas.registry import resolve as resolve_schema_id
from .bus_schemas import BaseEnvelope
from .codec import OrionCodec
from .enforce import enforcer

logger = logging.getLogger("orion.bus.async")


class OrionBusAsync:
    """
    Async Redis bus client.
    """

    def __init__(
        self,
        url: str,
        *,
        enabled: bool = True,
        codec: Optional[OrionCodec] = None,
        enforce_catalog: bool | None = None,
    ):
        self.url = url
        self.enabled = enabled
        self.codec = codec or OrionCodec()
        self._redis: Optional[aioredis.Redis] = None
        if enforce_catalog is None:
            enforce_catalog = os.getenv("ORION_BUS_ENFORCE_CATALOG", "false").lower() == "true"
        enforcer.enforce = enforce_catalog

    async def connect(self) -> None:
        if not self.enabled:
            return
        if self._redis is None:
            self._redis = aioredis.from_url(self.url, decode_responses=False)
            await self._redis.ping()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("OrionBusAsync not connected. Call await connect().")
        return self._redis

    async def publish(self, channel: str, msg: BaseEnvelope | Dict[str, Any]) -> None:
        if not self.enabled:
            return
        enforcer.validate(channel)
        self._validate_payload(channel, msg)
        data = self.codec.encode(msg)  # bytes
        await self.redis.publish(channel, data)

    def _validate_payload(self, channel: str, msg: BaseEnvelope | Dict[str, Any]) -> None:
        entry = enforcer.entry_for(channel)
        if not entry:
            return
        schema_id = entry.get("schema_id")
        if not schema_id:
            return
        payload = None
        if isinstance(msg, BaseEnvelope):
            payload = msg.payload
        elif isinstance(msg, dict) and msg.get("schema") == "orion.envelope":
            try:
                env = BaseEnvelope.model_validate(msg)
            except ValidationError as exc:
                raise ValueError(f"Envelope validation failed for channel {channel}") from exc
            payload = env.payload
        if payload is None:
            return
        # Boundary rule: payloads on the bus must be JSON-ish.
        # If a producer passes a Pydantic model, normalize to a dict before validation.
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")

        model = resolve_schema_id(schema_id)
        try:
            model.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Payload validation failed for channel {channel} (schema_id={schema_id})"
            ) from exc

    @asynccontextmanager
    async def subscribe(self, *channels: str, patterns: bool = False) -> AsyncIterator[aioredis.client.PubSub]:
        if not self.enabled:
            raise RuntimeError("Bus disabled")
        pubsub = self.redis.pubsub()
        if patterns:
            await pubsub.psubscribe(*channels)
        else:
            await pubsub.subscribe(*channels)
        try:
            yield pubsub
        finally:
            try:
                if patterns:
                    await pubsub.punsubscribe(*channels)
                else:
                    await pubsub.unsubscribe(*channels)
            finally:
                await pubsub.close()

    async def iter_messages(self, pubsub: aioredis.client.PubSub) -> AsyncIterator[dict]:
        """
        Unified async message iterator. Yields dicts with fields similar to redis-py's listen().
        """
        async for msg in pubsub.listen():
            mtype = msg.get("type")
            if mtype not in ("message", "pmessage"):
                continue
            yield msg

    async def rpc_request(
        self,
        request_channel: str,
        envelope: BaseEnvelope,
        *,
        reply_channel: str,
        timeout_sec: float = 60.0,
    ) -> dict:
        """
        Publish `envelope` to request_channel and await first message on reply_channel.
        """
        async with self.subscribe(reply_channel) as pubsub:
            await self.publish(request_channel, envelope)
            try:
                async def _wait_one():
                    async for msg in self.iter_messages(pubsub):
                        return msg
                msg = await asyncio.wait_for(_wait_one(), timeout=timeout_sec)
                return msg
            except asyncio.TimeoutError:
                raise TimeoutError(f"RPC timeout waiting on {reply_channel}")

    async def rpc_legacy_dict(
        self,
        request_channel: str,
        payload: Dict[str, Any],
        *,
        reply_prefix: str,
        timeout_sec: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Compatibility RPC helper for legacy dict-style services.

        Convention:
          - caller publishes a dict that includes `reply_channel` and `correlation_id`
          - callee replies with either a Titanium envelope OR a legacy json dict
        """
        if not self.enabled:
            raise RuntimeError("Bus disabled")

        from uuid import uuid4
        corr = uuid4()
        reply_channel = f"{reply_prefix}{corr}"

        msg_payload = dict(payload)
        msg_payload.setdefault("reply_channel", reply_channel)
        msg_payload.setdefault("correlation_id", str(corr))

        async with self.subscribe(reply_channel) as pubsub:
            await self.publish(request_channel, msg_payload)
            try:
                async def _wait_one():
                    async for msg in self.iter_messages(pubsub):
                        return msg
                msg = await asyncio.wait_for(_wait_one(), timeout=timeout_sec)
            except asyncio.TimeoutError as te:
                raise TimeoutError(f"RPC timeout waiting on {reply_channel}") from te

        decoded = self.codec.decode(msg.get("data"))
        if decoded.ok and isinstance(decoded.envelope.payload, dict):
            return decoded.envelope.payload

        # legacy fallback: raw json dict
        data = msg.get("data")
        if isinstance(data, (bytes, bytearray)):
            try:
                import json

                return json.loads(data.decode("utf-8", "ignore"))
            except Exception:
                pass
        return {"ok": False, "error": decoded.error or "decode_failed"}
