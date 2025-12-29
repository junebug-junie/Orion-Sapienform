# services/orion-agent-chain/app/planner_rpc.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict
from uuid import uuid4

from fastapi import HTTPException

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from .settings import settings

logger = logging.getLogger("agent-chain.rpc")

async def call_planner_react(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async Bus RPC to the planner-react service.
    Sends a strictly typed BaseEnvelope.
    """
    if not settings.orion_bus_enabled:
        raise HTTPException(
            status_code=500, 
            detail="Bus disabled; agent-chain cannot reach planner-react."
        )

    # Ephemeral connection (safe for both HTTP and Bus contexts)
    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()

    try:
        trace_id = payload.get("request_id") or str(uuid4())
        request_channel = settings.planner_request_channel
        reply_channel = f"{settings.planner_result_prefix}:{trace_id}"

        # 1. Construct Envelope
        # This removes the "wrapper" dict problem. We send the payload directly.
        env = BaseEnvelope(
            kind="agent.planner.request",
            source=ServiceRef(
                name=settings.service_name,
                version="0.2.0",
                node=getattr(settings, "node_name", "unknown")
            ),
            correlation_id=trace_id,
            reply_to=reply_channel,
            payload=payload,
        )

        logger.info(f"[RPC] -> {request_channel} (trace={trace_id})")

        # 2. Determine Timeout
        limits = payload.get("limits", {})
        timeout = float(limits.get("timeout_seconds", 60.0))

        # 3. Execute
        msg = await bus.rpc_request(
            request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout
        )

        # 4. Decode & Validate
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.error(f"Planner decode error: {decoded.error}")
            return {
                "status": "error",
                "error": {"message": f"Decode failed: {decoded.error}"}
            }

        # The Planner now returns a PlannerResponse (as a dict) in the payload
        response = decoded.envelope.payload
        if not isinstance(response, dict):
             # Fallback for safety
             return {
                 "status": "error",
                 "error": {"message": f"Invalid response type: {type(response)}"}
             }

        return response

    except asyncio.TimeoutError:
        logger.warning(f"Planner timeout (trace={trace_id})")
        return {
            "status": "timeout",
            "error": {"message": "planner-react timed out"},
            "request_id": trace_id
        }
    except Exception as e:
        logger.exception(f"RPC Unexpected Error: {e}")
        return {
            "status": "error",
            "error": {"message": str(e)},
            "request_id": trace_id
        }
    finally:
        await bus.close()
