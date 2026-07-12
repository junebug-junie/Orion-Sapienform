from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionRequest

logger = logging.getLogger("orion.harness.cortex")


def _clamp_plan_to_outer_timeout(req: PlanExecutionRequest, *, timeout_sec: float) -> PlanExecutionRequest:
    """Keep Cortex step waits below the harness bus RPC wait.

    The harness waits on Cortex with `timeout_sec`; Cortex then waits on
    LLMGateway using the plan/step timeout. If the inner timeout is larger than
    the outer one, the user-visible turn can appear hung at the finalize
    boundary. Clamp the plan before emission so the callee fails first.
    """
    outer = max(1.0, float(timeout_sec))
    cap_ms = max(1000, int(max(1.0, outer - 5.0) * 1000))
    plan = req.plan
    step_updates = []
    changed = plan.timeout_ms > cap_ms
    for step in plan.steps:
        next_timeout = min(int(step.timeout_ms or cap_ms), cap_ms)
        changed = changed or next_timeout != step.timeout_ms
        step_updates.append(step.model_copy(update={"timeout_ms": next_timeout}))
    if not changed:
        return req

    metadata = dict(plan.metadata or {})
    metadata["outer_rpc_timeout_sec"] = f"{outer:.3f}"
    metadata["timeout_clamped_ms"] = str(cap_ms)
    clamped_plan = plan.model_copy(
        update={
            "timeout_ms": min(int(plan.timeout_ms or cap_ms), cap_ms),
            "steps": step_updates,
            "metadata": metadata,
        }
    )
    return req.model_copy(update={"plan": clamped_plan})


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
        req = _clamp_plan_to_outer_timeout(req, timeout_sec=timeout_sec)
        return await self.execute_plan(
            req,
            correlation_id=correlation_id,
            timeout_sec=timeout_sec,
        )
