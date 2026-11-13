"""The broker for Claude's tools in Hub chat, beside dig and the introspect client (Patch 1b).

Subscribes to orion:claude:tool:request (one request per Claude tool call, from the MCP server inside
orion-room-companion) and answers on orion:claude:tool:result:<correlation_id>. The decisions live in
orion/claude_tools/broker.py; this module only wires the bus, Redis and the readers. Claude's process
never runs here and never gets a shell here: the only input is a validated request.

Each request runs in its own task: a consent ask can take up to 40 s and must not hold the next call.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from orion.claude_tools.broker import BrokerConfig, ClaudeToolBroker, bus_audit_publisher, bus_consent_asker
from orion.claude_tools.readers import read_body_economy, read_prime_now
from orion.claude_tools.render import unavailable_text
from orion.claude_tools.transport import TOOL_REQUEST_KIND, TOOL_RESULT_KIND, TOOL_RESULT_PREFIX
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import run_subscription
from orion.schemas.claude_tools import ClaudeToolRequestV1, ClaudeToolResultV1

from .settings import settings

logger = logging.getLogger("orion-harness-governor.claude_tools")


def build_broker(bus: Any) -> ClaudeToolBroker:
    source = ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)
    return ClaudeToolBroker(
        config=BrokerConfig(
            enabled=settings.claude_tools_enabled,
            max_calls_per_turn=settings.claude_tools_max_calls_per_turn,
            max_asks_per_turn=settings.claude_tools_max_asks_per_turn,
            consent_deadline_sec=settings.claude_tools_consent_deadline_sec,
        ),
        redis=bus.redis,
        ask_orion=bus_consent_asker(bus, source),
        readers={
            "prime_now": lambda args: read_prime_now(bus.redis, args),
            "body_economy": read_body_economy,
        },
        publish_audit=bus_audit_publisher(bus, source),
    )


async def handle_envelope(bus: Any, broker: ClaudeToolBroker, envelope: BaseEnvelope) -> None:
    reply = f"{TOOL_RESULT_PREFIX}{envelope.correlation_id}"
    if envelope.reply_to != reply or envelope.kind != TOOL_REQUEST_KIND:
        return
    source = ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)
    try:
        request = ClaudeToolRequestV1.model_validate(envelope.payload)
    except ValueError as exc:
        payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        result = ClaudeToolResultV1(
            call_id=str(payload.get("call_id") or "unknown")[:64], tool=str(payload.get("tool") or "unknown")[:64],
            shared=False, decision_path="invalid", text=unavailable_text("the tool request was malformed"),
            error=str(exc)[:500],
        )
        logger.warning("claude_tool_bad_request corr=%s err=%s", envelope.correlation_id, str(exc)[:200])
    else:
        result = await broker.handle(request)
    await bus.publish(reply, BaseEnvelope(kind=TOOL_RESULT_KIND, correlation_id=envelope.correlation_id,
                                          source=source, payload=result.model_dump(mode="json")))


async def run_claude_tools_worker(stop_event: Any | None = None) -> None:
    if not settings.orion_bus_enabled:
        logger.info("claude tools broker idle (bus disabled)")
        return
    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()
    broker = build_broker(bus)
    inflight: set[asyncio.Task] = set()
    logger.info("claude_tools_broker_listening channel=%s enabled=%s", settings.channel_claude_tool_request,
                settings.claude_tools_enabled)

    async def handle(msg: dict[str, Any]) -> None:
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.warning("claude_tool_decode_failed err=%s", decoded.error)
            return
        task = asyncio.create_task(_guarded(bus, broker, decoded.envelope))
        inflight.add(task)
        task.add_done_callback(inflight.discard)

    try:
        await run_subscription(bus, settings.channel_claude_tool_request, handle, stop_event=stop_event,
                               name="harness-governor-claude-tools")
    except asyncio.CancelledError:
        raise
    finally:
        for task in list(inflight):
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        with suppress(Exception):
            await bus.close()


async def _guarded(bus: Any, broker: ClaudeToolBroker, envelope: BaseEnvelope) -> None:
    try:
        await handle_envelope(bus, broker, envelope)
    except Exception:  # noqa: BLE001 -- one bad call must not kill the worker; the MCP side times out to "not shared"
        logger.exception("claude_tool_handle_failed corr=%s", envelope.correlation_id)


__all__ = ["build_broker", "handle_envelope", "run_claude_tools_worker"]
