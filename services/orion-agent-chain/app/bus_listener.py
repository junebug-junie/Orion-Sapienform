from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Dict
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.agents.schemas import AgentChainRequest

from .api import execute_agent_chain
from .settings import settings

logger = logging.getLogger("agent-chain.bus")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=getattr(settings, "node_name", None),
        version=settings.service_version,
    )


async def run_bus_worker(stop_event: asyncio.Event | None = None) -> None:
    """Long-running bus consumer for AgentChainService RPC requests."""
    if not settings.orion_bus_enabled:
        logger.info("[agent-chain] Bus disabled; worker not started")
        return

    bus = OrionBusAsync(url=settings.orion_bus_url)
    request_channel = settings.agent_chain_request_channel
    accepted_kinds = ("agent.chain.request", "legacy.message")

    logger.info("[agent-chain] bus connected url=%s", settings.orion_bus_url)
    await bus.connect()
    logger.info(
        "[agent-chain] subscribed channel=%s kinds=%s",
        request_channel,
        accepted_kinds,
    )

    try:
        async with bus.subscribe(request_channel) as pubsub:
            while True:
                if stop_event is not None and stop_event.is_set():
                    logger.info("[agent-chain] stop event set; exiting bus worker")
                    break
                try:
                    msg = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0), timeout=1.2)
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    await _handle_request(bus, msg)
                except Exception:
                    logger.exception("[agent-chain] unhandled worker error while processing message")
    except asyncio.CancelledError:
        logger.info("[agent-chain] bus worker cancellation requested")
        raise
    finally:
        with suppress(Exception):
            await bus.close()
        logger.info("[agent-chain] bus worker stopped")


async def _handle_request(bus: OrionBusAsync, raw_msg: Dict[str, Any]) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("[agent-chain] Decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    reply_channel = env.reply_to
    payload = env.payload or {}
    if not reply_channel:
        reply_channel = payload.get("reply_channel")

    if env.kind and env.kind not in ("agent.chain.request", "legacy.message"):
        logger.warning("[agent-chain] Unsupported kind=%s", env.kind)
        return

    incoming_corr = str(env.correlation_id) if getattr(env, "correlation_id", None) else str(payload.get("request_id") or uuid4())
    logger.info("[agent-chain] intake parent=%s reply_to=%s", incoming_corr, reply_channel)
    logger.info("[agent-chain] received kind=%s corr_id=%s", env.kind, incoming_corr)

    if not reply_channel:
        logger.debug("[agent-chain] No reply channel, ignoring message.")
        return

    try:
        req = AgentChainRequest(**payload)
        rpc_bus = await bus.fork(start_rpc_worker=False)
        logger.info("[agent-chain] planner rpc bus=fork parent=%s rpc_worker=%s", incoming_corr, False)
        result = await execute_agent_chain(req, correlation_id=incoming_corr, rpc_bus=rpc_bus)
        resp = BaseEnvelope(
            kind="agent.chain.result",
            source=_source(),
            correlation_id=incoming_corr,
            causality_chain=env.causality_chain,
            payload=result.model_dump(mode="json"),
        )
        logger.info("[agent-chain] replying to exec parent=%s reply_to=%s", incoming_corr, reply_channel)
        await bus.publish(reply_channel, resp)
        logger.info("[agent-chain] replied reply_to=%s corr_id=%s kind=%s", reply_channel, incoming_corr, resp.kind)
    except Exception as e:
        logger.error("[agent-chain] Execution Error: %s", e)
        error_payload = {
            "mode": payload.get("mode") or "agent",
            "text": f"Agent-chain error: {e}",
            "structured": {},
            "planner_raw": {"status": "error", "error": str(e)},
        }
        error_env = BaseEnvelope(
            kind="agent.chain.result",
            source=_source(),
            correlation_id=incoming_corr,
            causality_chain=env.causality_chain,
            payload=error_payload,
        )
        logger.info("[agent-chain] replying to exec parent=%s reply_to=%s", incoming_corr, reply_channel)
        try:
            await bus.publish(reply_channel, error_env)
            logger.info("[agent-chain] replied reply_to=%s corr_id=%s kind=%s", reply_channel, incoming_corr, error_env.kind)
        except Exception:
            logger.exception("[agent-chain] failed to publish error response parent=%s reply_to=%s", incoming_corr, reply_channel)
    finally:
        if 'rpc_bus' in locals():
            with suppress(Exception):
                await rpc_bus.close()
