# services/orion-planner-react/app/bus_listener.py
from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from .settings import settings
from .api import run_react_loop
from orion.schemas.agents.schemas import PlannerRequest, PlannerResponse

logger = logging.getLogger("planner-react.bus")

def start_planner_bus_listener_background() -> None:
    if not settings.orion_bus_enabled:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop to attach bus listener.")
        return

    loop.create_task(_async_bus_worker())

async def _async_bus_worker() -> None:
    bus = OrionBusAsync(url=settings.orion_bus_url)
    request_channel = settings.planner_request_channel
    
    logger.info("[planner-react] Connecting Async Bus listener on %s...", request_channel)
    await bus.connect()

    async with bus.subscribe(request_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            asyncio.create_task(_handle_request(bus, msg))

async def _handle_request(bus: OrionBusAsync, raw_msg: Dict[str, Any]) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        return

    env = decoded.envelope
    reply_channel = env.reply_to
    
    # Payload Logic: 
    # 1. BaseEnvelope puts the data in env.payload
    payload = env.payload or {}
    
    # 2. Legacy Protection: If the payload is STILL a wrapper (has 'event' and nested 'payload')
    # we unwrap it. This protects against other services (not just agent-chain) doing it wrong.
    if "payload" in payload and "event" in payload:
        payload = payload["payload"]

    # 3. Fallback for legacy routing
    if not reply_channel:
        reply_channel = payload.get("reply_channel") or env.payload.get("reply_channel")

    trace_id = env.correlation_id or payload.get("request_id")

    if not reply_channel:
        return

    try:
        # STRICT VALIDATION
        planner_req = PlannerRequest(**payload)
        
        # EXECUTE
        resp = await run_react_loop(planner_req)
        
        if resp.request_id is None:
            resp.request_id = planner_req.request_id or str(trace_id)

        await bus.publish(reply_channel, resp.model_dump(mode="json"))

        logger.info(
            "[planner-react] Finished trace=%s status=%s steps=%d",
            trace_id, resp.status, (resp.usage.steps if resp.usage else 0)
        )

    except Exception as e:
        logger.error("[planner-react] Error processing %s: %s", trace_id, e)
        error_resp = PlannerResponse(
            request_id=str(trace_id),
            status="error",
            error={"message": str(e)}
        )
        await bus.publish(reply_channel, error_resp.model_dump(mode="json"))
