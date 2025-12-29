# services/orion-cortex-orch/app/clients.py
from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionRequest

logger = logging.getLogger("orion.cortex.orch.clients")

class CortexExecClient:
    """
    Strict, typed client for sending plans to cortex-exec.
    """
    def __init__(self, bus: OrionBusAsync, channel: str):
        self.bus = bus
        self.channel = channel

    async def execute_plan(
        self,
        source: ServiceRef,
        req: PlanExecutionRequest,
        correlation_id: str,
        timeout_sec: float
    ) -> Dict[str, Any]:
        """
        Sends a typed PlanExecutionRequest, returns the raw result dict from Exec.
        """
        reply_channel = f"orion-cortex-exec:result:{uuid4()}"
        
        # 1. STRICT: Convert Pydantic -> JSON
        # This ensures we never send a malformed plan
        payload_json = req.model_dump(mode="json")

        env = BaseEnvelope(
            kind="cortex.exec.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=payload_json,
        )

        logger.info(f"Sending Plan to {self.channel} (steps={len(req.plan.steps)})")

        # 2. TRANSPORT
        msg = await self.bus.rpc_request(
            self.channel, 
            env, 
            reply_channel=reply_channel, 
            timeout_sec=timeout_sec
        )

        # 3. DECODE
        decoded = self.bus.codec.decode(msg.get("data"))
        
        if not decoded.ok:
            raise RuntimeError(f"Exec RPC failed: {decoded.error}")
        
        if not isinstance(decoded.envelope.payload, dict):
             # Just in case Exec sends back something weird
             return decoded.envelope.model_dump(mode="json")
             
        return decoded.envelope.payload
