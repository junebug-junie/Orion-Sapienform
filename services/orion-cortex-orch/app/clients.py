# services/orion-cortex-orch/app/clients.py
from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

from pydantic import ValidationError
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.exec import CortexExecResultPayload
from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply

logger = logging.getLogger("orion.cortex.orch.clients")

class CortexExecClient:
    """
    Strict, typed client for sending plans to cortex-exec.
    """
    def __init__(self, bus: OrionBusAsync, *, request_channel: str, result_prefix: str):
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix

    async def execute_plan(
        self,
        source: ServiceRef,
        req: PlanExecutionRequest,
        correlation_id: str,
        timeout_sec: float,
        *,
        trace: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Sends a typed PlanExecutionRequest, returns the raw result dict from Exec.
        """
        reply_channel = f"{self.result_prefix}:{uuid4()}"

        # 1. STRICT: Convert Pydantic -> JSON
        # This ensures we never send a malformed plan
        payload_json = req.model_dump(mode="json")

        env = BaseEnvelope(
            kind=req.kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            trace=dict(trace or {}),
            payload=payload_json,
        )

        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s steps=%s",
            self.request_channel,
            env.kind,
            correlation_id,
            reply_channel,
            len(req.plan.steps),
        )

        # 2. TRANSPORT
        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec
        )

        # 3. DECODE
        decoded = self.bus.codec.decode(msg.get("data"))

        if not decoded.ok:
            raise RuntimeError(f"Exec RPC failed: {decoded.error}")

        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            try:
                exec_payload = CortexExecResultPayload.model_validate(payload)
            except ValidationError:
                return payload.get("result") or payload
            if exec_payload.result is not None:
                return exec_payload.result.model_dump(mode="json")
            if exec_payload.error:
                return {"ok": False, "error": exec_payload.error, "details": exec_payload.details}
            return payload

        # Just in case Exec sends back something weird
        return decoded.envelope.model_dump(mode="json")


class StateServiceClient:
    """
    Strict, typed client for requesting the latest Orion state (Spark snapshot)
    from orion-state-service.
    """
    def __init__(self, bus: OrionBusAsync, *, request_channel: str, result_prefix: str):
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix

    async def get_latest(
        self,
        *,
        source: ServiceRef,
        req: StateGetLatestRequest,
        correlation_id: str,
        timeout_sec: float
    ) -> StateLatestReply:
        reply_channel = f"{self.result_prefix}:{uuid4()}"

        env = BaseEnvelope(
            kind=req.kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )

        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"State RPC decode failed: {decoded.error}")

        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            return StateLatestReply.model_validate(payload)

        # Fallback: try to validate whole envelope
        return StateLatestReply.model_validate(decoded.envelope.model_dump(mode="json").get("payload") or {})
