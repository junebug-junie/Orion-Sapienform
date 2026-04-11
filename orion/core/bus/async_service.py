# orion/core/bus/async_service.py
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from time import perf_counter
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
        self._rpc_pubsub: Optional[aioredis.client.PubSub] = None
        self._rpc_worker_task: Optional[asyncio.Task] = None
        self._rpc_worker_running = False
        self._rpc_lock = asyncio.Lock()
        self._rpc_subscribed: set[str] = set()
        self._pending_rpc: dict[tuple[str, str], asyncio.Future] = {}
        if enforce_catalog is None:
            enforce_catalog = os.getenv("ORION_BUS_ENFORCE_CATALOG", "false").lower() == "true"
        self.enforce_catalog = bool(enforce_catalog)
        enforcer.enforce = enforce_catalog

    async def fork(self, *, start_rpc_worker: bool = False) -> "OrionBusAsync":
        """
        Create an independent bus client sharing config/codec with a fresh Redis connection.

        Useful for nested RPC in services that already maintain a long-lived subscriber on
        another bus instance.
        """
        child = OrionBusAsync(
            url=self.url,
            enabled=self.enabled,
            codec=self.codec,
            enforce_catalog=self.enforce_catalog,
        )
        if start_rpc_worker:
            await child.connect()
            child.start_rpc_worker()
        return child

    def start_rpc_worker(self) -> None:
        if not self.enabled:
            return
        if self._rpc_worker_task and not self._rpc_worker_task.done():
            return
        self._rpc_worker_running = True
        self._rpc_worker_task = asyncio.create_task(self._run_rpc_only())
        logger.info("[rpc-fork] worker started")

    async def _run_rpc_only(self) -> None:
        if not self.enabled:
            return
        await self.connect()
        self._rpc_pubsub = self.redis.pubsub()
        try:
            while self._rpc_worker_running:
                msg = await self._rpc_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                await self._handle_rpc_result(msg)
        except asyncio.CancelledError:
            raise
        finally:
            if self._rpc_pubsub is not None:
                with suppress(Exception):
                    if self._rpc_subscribed:
                        await self._rpc_pubsub.unsubscribe(*sorted(self._rpc_subscribed))
                with suppress(Exception):
                    await self._rpc_pubsub.close()
                self._rpc_pubsub = None
            self._rpc_worker_running = False

    async def _handle_rpc_result(self, msg: dict) -> None:
        channel_raw = msg.get("channel")
        channel = channel_raw.decode("utf-8", "ignore") if isinstance(channel_raw, (bytes, bytearray)) else str(channel_raw)
        decoded = self.codec.decode(msg.get("data"))
        if not decoded.ok:
            return
        corr = str(getattr(decoded.envelope, "correlation_id", "") or "")
        logger.info("[rpc-fork] reply received corr_id=%s reply_channel=%s", corr, channel)
        key = (channel, corr)
        fut = self._pending_rpc.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(msg)
            logger.info("[rpc-fork] future resolved corr_id=%s reply_channel=%s", corr, channel)

    async def _rpc_subscribe(self, reply_channel: str) -> None:
        await self.connect()
        if self._rpc_pubsub is None and self._rpc_worker_running:
            for _ in range(50):
                if self._rpc_pubsub is not None:
                    break
                await asyncio.sleep(0.01)
        if self._rpc_pubsub is None:
            raise RuntimeError("RPC worker pubsub is not initialized")
        if reply_channel in self._rpc_subscribed:
            logger.info("[rpc] reply-listener already subscribed reply_channel=%s", reply_channel)
            return
        logger.info("[rpc] reply-listener creating reply_channel=%s", reply_channel)
        await self._rpc_pubsub.subscribe(reply_channel)
        self._rpc_subscribed.add(reply_channel)
        logger.info("[rpc] reply-listener ready reply_channel=%s", reply_channel)

    async def connect(self) -> None:
        if not self.enabled:
            return
        if self._redis is None:
            self._redis = aioredis.from_url(self.url, decode_responses=False)
            await self._redis.ping()

    async def close(self) -> None:
        self._rpc_worker_running = False
        if self._rpc_worker_task is not None:
            self._rpc_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._rpc_worker_task
            self._rpc_worker_task = None
        for fut in self._pending_rpc.values():
            if not fut.done():
                fut.cancel()
        self._pending_rpc.clear()
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
        started = perf_counter()
        corr = str(envelope.correlation_id)
        async with self.subscribe(reply_channel) as pubsub:
            try:
                if self._rpc_worker_task and not self._rpc_worker_task.done():
                    key = (reply_channel, corr)
                    fut = asyncio.get_running_loop().create_future()
                    self._pending_rpc[key] = fut
                    async with self._rpc_lock:
                        await self._rpc_subscribe(reply_channel)
                    logger.info(
                        "[rpc] publish begin corr_id=%s request_channel=%s reply_channel=%s path=worker elapsed_ms=%.1f",
                        corr,
                        request_channel,
                        reply_channel,
                        (perf_counter() - started) * 1000.0,
                    )
                    await self.publish(request_channel, envelope)
                    logger.info(
                        "[rpc] publish success corr_id=%s request_channel=%s reply_channel=%s path=worker elapsed_ms=%.1f",
                        corr,
                        request_channel,
                        reply_channel,
                        (perf_counter() - started) * 1000.0,
                    )
                    try:
                        logger.info(
                            "[rpc] waiting for reply corr_id=%s reply_channel=%s timeout_sec=%.2f path=worker",
                            corr,
                            reply_channel,
                            timeout_sec,
                        )
                        return await asyncio.wait_for(fut, timeout=timeout_sec)
                    finally:
                        self._pending_rpc.pop(key, None)

                logger.info(
                    "[rpc] publish begin corr_id=%s request_channel=%s reply_channel=%s path=inline elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    (perf_counter() - started) * 1000.0,
                )
                await self.publish(request_channel, envelope)
                logger.info(
                    "[rpc] publish success corr_id=%s request_channel=%s reply_channel=%s path=inline elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    (perf_counter() - started) * 1000.0,
                )

                async def _wait_one():
                    async for msg in self.iter_messages(pubsub):
                        logger.info(
                            "[rpc] reply received corr_id=%s reply_channel=%s path=inline elapsed_ms=%.1f",
                            corr,
                            reply_channel,
                            (perf_counter() - started) * 1000.0,
                        )
                        return msg

                logger.info(
                    "[rpc] waiting for reply corr_id=%s reply_channel=%s timeout_sec=%.2f path=inline",
                    corr,
                    reply_channel,
                    timeout_sec,
                )
                msg = await asyncio.wait_for(_wait_one(), timeout=timeout_sec)
                return msg
            except asyncio.TimeoutError:
                logger.error(
                    "[rpc] timeout waiting for reply corr_id=%s request_channel=%s reply_channel=%s timeout_sec=%.2f elapsed_ms=%.1f",
                    corr,
                    request_channel,
                    reply_channel,
                    timeout_sec,
                    (perf_counter() - started) * 1000.0,
                )
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
