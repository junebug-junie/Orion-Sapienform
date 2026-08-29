from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef, SYSTEM_ERROR_KINDS
from orion.schemas.cortex.schemas import PlanExecutionRequest

logger = logging.getLogger("orion.harness.cortex")


class HarnessCortexClient:
    """Thin RPC client for cortex-exec finalize verbs (5b / 5c)."""

    def __init__(
        self,
        bus: OrionBusAsync,
        *,
        request_channel: str,
        result_prefix: str,
        source_name: str = "orion-harness-governor",
        timeout_sec: float = 15.0,
        voice_finalize_timeout_sec: float | None = None,
    ) -> None:
        self.bus = bus
        self.request_channel = request_channel
        self.result_prefix = result_prefix
        self.source_name = source_name
        self.timeout_sec = timeout_sec
        self.voice_finalize_timeout_sec = voice_finalize_timeout_sec

    async def execute_plan(
        self,
        req: PlanExecutionRequest,
        *,
        correlation_id: str,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        reply_channel = f"{self.result_prefix}:{uuid4()}"
        source = ServiceRef(name=self.source_name)
        env = BaseEnvelope(
            kind=req.kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )
        logger.info(
            "harness cortex RPC -> %s verb=%s corr=%s",
            self.request_channel,
            req.plan.verb_name,
            correlation_id,
        )
        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec if timeout_sec is not None else self.timeout_sec,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Harness cortex RPC failed: {decoded.error}")

        payload = decoded.envelope.payload
        if decoded.envelope.kind in SYSTEM_ERROR_KINDS:
            # Without this check, a system.error reply (no "result" key) fell
            # through to `return payload` below and was handed to the caller
            # as if it were real plan-execution output -- silent data
            # corruption, not a raised error. Confirmed live 2026-08-29: this
            # channel's Rabbit chassis (orion/core/bus/bus_service_chassis.py)
            # publishes exactly this shape on any handler exception.
            detail = payload.get("error") if isinstance(payload, dict) else None
            # `is not None`, not `or` -- a producer-sent empty-string error
            # must not fall through to dumping the whole payload instead.
            raise RuntimeError(f"Harness cortex RPC returned an error: {detail if detail is not None else payload}")
        if isinstance(payload, dict):
            nested = payload.get("result")
            if isinstance(nested, dict):
                return nested
            return payload
        raise RuntimeError("Harness cortex RPC returned non-dict payload")

    async def __call__(self, req: PlanExecutionRequest) -> dict[str, Any]:
        correlation_id = str(req.args.request_id or uuid4())
        verb = req.plan.verb_name
        if verb == "orion_voice_finalize":
            timeout_sec = self.voice_finalize_timeout_sec or self.timeout_sec
        else:
            timeout_sec = self.timeout_sec
        return await self.execute_plan(
            req,
            correlation_id=correlation_id,
            timeout_sec=timeout_sec,
        )
