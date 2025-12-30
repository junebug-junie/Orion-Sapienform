# services/orion-agent-chain/app/bus_listener.py
from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from .settings import settings
from .api import execute_agent_chain
from orion.schemas.agents.schemas import AgentChainRequest

logger = logging.getLogger("agent-chain.bus")

def start_agent_chain_bus_listener() -> None:
    if not settings.orion_bus_enabled:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    loop.create_task(_async_bus_worker())

async def _async_bus_worker() -> None:
    bus = OrionBusAsync(url=settings.orion_bus_url)
    request_channel = settings.agent_chain_request_channel

    logger.info("[agent-chain] Connecting Async Bus on %s", request_channel)
    await bus.connect()

    # Use Context Manager correctly
    async with bus.subscribe(request_channel) as pubsub:
        # Iterate using the helper
        async for msg in bus.iter_messages(pubsub):
            asyncio.create_task(_handle_request(bus, msg))


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=getattr(settings, "node_name", None),
        version=settings.service_version,
    )

async def _handle_request(bus: OrionBusAsync, raw_msg: Dict[str, Any]) -> None:
    # Decode raw bytes
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning(f"[agent-chain] Decode failed: {decoded.error}")
        return

    env = decoded.envelope
    reply_channel = env.reply_to

    # Handle legacy payloads where reply_channel might be inside the dict
    payload = env.payload or {}
    if not reply_channel:
        reply_channel = payload.get("reply_channel")

    if env.kind and env.kind not in ("agent.chain.request", "legacy.message"):
        logger.warning("[agent-chain] Unsupported kind=%s", env.kind)
        return

    trace_id = env.correlation_id or payload.get("request_id")

    if not reply_channel:
        logger.debug("[agent-chain] No reply channel, ignoring message.")
        return

    try:
        # Strict Shared Schema
        req = AgentChainRequest(**payload)

        result = await execute_agent_chain(req)
        resp = BaseEnvelope(
            kind="agent.chain.result",
            source=_source(),
            correlation_id=trace_id,
            causality_chain=env.causality_chain,
            payload=result.model_dump(mode="json"),
        )
        await bus.publish(reply_channel, resp)

    except Exception as e:
        logger.error("[agent-chain] Execution Error: %s", e)
        error_env = BaseEnvelope(
            kind="agent.chain.result",
            source=_source(),
            correlation_id=trace_id,
            causality_chain=env.causality_chain,
            payload={"error": str(e)},
        )
        await bus.publish(reply_channel, error_env)
