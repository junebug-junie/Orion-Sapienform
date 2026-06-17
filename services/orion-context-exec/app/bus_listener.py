from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Dict
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.agents.schemas import AgentChainRequest

from .agent_compat import agent_chain_request_to_context_exec, context_exec_run_to_agent_chain_result
from .runner import ContextExecRunner
from .settings import settings

logger = logging.getLogger("orion-context-exec.bus")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


async def run_bus_worker(stop_event: asyncio.Event | None = None) -> None:
    if not settings.orion_bus_enabled:
        logger.info("Bus disabled; worker not started")
        return

    bus = OrionBusAsync(url=settings.orion_bus_url)
    channels = [settings.channel_context_exec_intake]
    if settings.context_exec_compat_agent_chain_enabled:
        alias = settings.context_exec_agent_chain_intake_alias
        if alias not in channels:
            channels.append(alias)

    await bus.connect()
    from orion.core.bus.rpc_fork import fork_rpc_client

    rpc_bus = await fork_rpc_client(bus)
    logger.info("subscribed channels=%s compat_alias=%s", channels, settings.context_exec_compat_agent_chain_enabled)
    runner = ContextExecRunner(bus=bus, rpc_bus=rpc_bus)

    try:
        async with bus.subscribe(*channels) as pubsub:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=1.2,
                    )
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    await _handle_request(bus, msg, runner)
                except Exception:
                    logger.exception("unhandled bus worker error")
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(Exception):
            await rpc_bus.close()
        with suppress(Exception):
            await bus.close()


async def _handle_request(bus: OrionBusAsync, raw_msg: Dict[str, Any], runner: ContextExecRunner) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    reply_channel = env.reply_to or (env.payload or {}).get("reply_channel")
    if not reply_channel:
        return

    kind = env.kind or ""
    compat = kind == "agent.chain.request"
    if kind not in ("context.exec.request.v1", "agent.chain.request", "legacy.message"):
        logger.warning("unsupported kind=%s", kind)
        return

    corr = str(env.correlation_id or uuid4())
    payload = env.payload or {}
    causality = list(env.causality_chain or [])

    try:
        if compat:
            body = AgentChainRequest(**payload)
            req = agent_chain_request_to_context_exec(body)
            if not req.correlation_id:
                req = req.model_copy(update={"correlation_id": corr})
            run = await runner.run(req, causality_chain=causality)
            result = context_exec_run_to_agent_chain_result(run, mode=body.mode or "agent")
            resp_kind = "agent.chain.result"
            resp_payload = result.model_dump(mode="json")
        else:
            from orion.schemas.context_exec import ContextExecRequestV1

            req = ContextExecRequestV1.model_validate(payload)
            if not req.correlation_id:
                req = req.model_copy(update={"correlation_id": corr})
            run = await runner.run(req, causality_chain=causality)
            resp_kind = "context.exec.result.v1"
            resp_payload = run.model_dump(mode="json")

        resp = BaseEnvelope(
            kind=resp_kind,
            source=_source(),
            correlation_id=corr,
            causality_chain=causality,
            payload=resp_payload,
        )
        await bus.publish(reply_channel, resp)
        logger.info("replied kind=%s corr=%s reply=%s", resp_kind, corr, reply_channel)
    except Exception as exc:
        logger.error("execution error corr=%s err=%s", corr, exc)
        err_payload = {
            "mode": payload.get("mode") or "agent",
            "text": f"Context-exec error: {exc}",
            "structured": {"context_exec_error": str(exc)},
            "planner_raw": {"runtime_debug": {"engine": "context_exec", "error": str(exc)}},
        }
        await bus.publish(
            reply_channel,
            BaseEnvelope(
                kind="agent.chain.result" if compat else "context.exec.result.v1",
                source=_source(),
                correlation_id=corr,
                causality_chain=causality,
                payload=err_payload,
            ),
        )
