# services/orion-agent-chain/app/planner_rpc.py

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict

from fastapi import HTTPException

from orion.core.bus.service import OrionBus
from .settings import settings


async def call_planner_react(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bus-native RPC to the planner-react service.

    Expects `payload` to match PlannerRequest JSON shape.
    """
    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    if not bus.enabled:
        raise HTTPException(
            status_code=500,
            detail="OrionBus is disabled; agent-chain cannot reach planner-react.",
        )

    trace_id = str(uuid.uuid4())
    request_channel = settings.planner_request_channel
    reply_channel = f"{settings.planner_result_prefix}:{trace_id}"

    envelope: Dict[str, Any] = {
        "event": "plan_react",
        "trace_id": trace_id,
        "origin_node": settings.service_name,
        "reply_channel": reply_channel,
        "payload": payload,
    }

    sub = bus.raw_subscribe(reply_channel)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def listener():
        try:
            for msg in sub:
                loop.call_soon_threadsafe(queue.put_nowait, msg)
                break
        finally:
            # raw_subscribe closes pubsub itself
            pass

    asyncio.get_running_loop().run_in_executor(None, listener)

    # Publish AFTER subscription is live
    bus.publish(request_channel, envelope)

    try:
        msg = await asyncio.wait_for(
            queue.get(),
            timeout=float(payload.get("limits", {}).get("timeout_seconds", 60)),
        )
        data = msg.get("data") or {}
        return data
    except asyncio.TimeoutError:
        return {
            "request_id": payload.get("request_id"),
            "status": "timeout",
            "error": {"message": "planner-react timed out"},
            "final_answer": None,
            "trace": [],
            "usage": None,
        }

