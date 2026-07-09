from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.thought import StanceReactRequestV1, ThoughtEventV1
from scripts.settings import settings

logger = logging.getLogger("hub.bus.thought")

_THOUGHT_SUBSCRIBER_WAIT_ATTEMPTS = 3
_THOUGHT_SUBSCRIBER_WAIT_SEC = 1.0


@dataclass(frozen=True, slots=True)
class ThoughtReactResult:
    thought: ThoughtEventV1 | None = None
    failure_reason: str | None = None


async def _thought_request_subscribers(bus: OrionBusAsync, channel: str) -> int:
    """Return subscriber count for channel, or -1 when the probe itself fails."""
    try:
        pairs = await bus.redis.pubsub_numsub(channel)
    except Exception as exc:  # noqa: BLE001 — probe must not fail the turn
        logger.warning("thought subscriber probe failed channel=%s err=%s", channel, exc)
        return -1
    for name, count in pairs:
        key = name.decode() if isinstance(name, bytes) else str(name)
        if key == channel:
            return int(count)
    return 0


async def _wait_for_thought_subscriber(
    bus: OrionBusAsync,
    channel: str,
    *,
    correlation_id: str,
) -> bool:
    saw_zero_subscribers = False
    for attempt in range(_THOUGHT_SUBSCRIBER_WAIT_ATTEMPTS):
        subs = await _thought_request_subscribers(bus, channel)
        if subs > 0:
            return True
        if subs < 0:
            logger.warning(
                "[%s] thought subscriber probe failed attempt=%s/%s; proceeding fail-open",
                correlation_id,
                attempt + 1,
                _THOUGHT_SUBSCRIBER_WAIT_ATTEMPTS,
            )
            return True
        saw_zero_subscribers = True
        logger.warning(
            "[%s] thought request channel has no subscriber attempt=%s/%s",
            correlation_id,
            attempt + 1,
            _THOUGHT_SUBSCRIBER_WAIT_ATTEMPTS,
        )
        if attempt + 1 < _THOUGHT_SUBSCRIBER_WAIT_ATTEMPTS:
            await asyncio.sleep(_THOUGHT_SUBSCRIBER_WAIT_SEC)
    return not saw_zero_subscribers


class ThoughtClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self._source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

    async def react(
        self,
        request: StanceReactRequestV1,
        *,
        correlation_id: Optional[str] = None,
        timeout_sec: float | None = None,
    ) -> ThoughtReactResult:
        correlation_id = correlation_id or request.correlation_id or str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_THOUGHT_RESULT_PREFIX}{correlation_id}"
        wait_sec = max(0.1, float(timeout_sec if timeout_sec is not None else settings.TIMEOUT_SEC))
        envelope = BaseEnvelope(
            kind="stance.react.request.v1",
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request.model_dump(mode="json"),
        )
        if not await _wait_for_thought_subscriber(
            self.bus,
            settings.CHANNEL_THOUGHT_REQUEST,
            correlation_id=correlation_id,
        ):
            logger.warning("[%s] thought RPC skipped: no subscriber on request channel", correlation_id)
            return ThoughtReactResult(failure_reason="thought_no_subscriber")
        try:
            msg = await self.bus.rpc_request(
                settings.CHANNEL_THOUGHT_REQUEST,
                envelope,
                reply_channel=reply_to,
                timeout_sec=wait_sec,
            )
        except TimeoutError:
            logger.warning("[%s] thought RPC timeout", correlation_id)
            return ThoughtReactResult(failure_reason="stance_react_timeout")
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return ThoughtReactResult(failure_reason="thought_rpc_decode_failed")
        if decoded.envelope.kind == "system.error":
            payload = decoded.envelope.payload
            detail = payload.get("details") if isinstance(payload, dict) else payload
            logger.warning("[%s] thought RPC system.error detail=%s", correlation_id, detail)
            return ThoughtReactResult(failure_reason="thought_rpc_system_error")
        payload = decoded.envelope.payload
        if isinstance(payload, dict) and payload.get("error"):
            logger.warning("[%s] thought RPC error payload=%s", correlation_id, payload.get("error"))
            return ThoughtReactResult(failure_reason="thought_rpc_error_payload")
        if isinstance(payload, dict):
            return ThoughtReactResult(thought=ThoughtEventV1.model_validate(payload))
        return ThoughtReactResult(failure_reason="thought_rpc_empty_payload")
