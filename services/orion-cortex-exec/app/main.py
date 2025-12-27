"""Cortex Exec (runtime) service.

This service is the *execution engine* in the full-fidelity pipeline:

  Hub -> Cortex Orch -> Cortex Exec -> (LLMGateway / RDF / ...) -> Exec -> Orch -> Hub

It exposes:

1) Bus-first RPC (preferred)
   - Intake:  CORTEX_EXEC_REQUEST_CHANNEL (default: orion-cortex-exec:request)
   - Reply:   reply_channel carried on the request (result_channel alias)

2) HTTP (debug / legacy)
   - POST /execute  (accepts a fully materialized ExecutionPlan)

Note:
- Exec should be the *only* component that talks to downstream service channels
  like `orion-exec:request:*`. Orch should call Exec, not fan-out directly.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
from orion.core.bus.service_async import OrionBusAsync

from .bus_models import CortexExecStepReplyV1, CortexExecStepRequestV1
from .executor import StepExecutor
from .models import PlanExecutionRequest, PlanExecutionResult
from .settings import settings

logger = logging.getLogger("orion-cortex-exec")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


app = FastAPI(title="orion-cortex-exec", version="0.1.0")

bus = OrionBusAsync(url=settings.bus_url, enabled=settings.bus_enabled)
executor = StepExecutor(bus)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.service_name,
        "node": settings.node_name,
        "bus_enabled": settings.bus_enabled,
    }


@app.post("/execute", response_model=PlanExecutionResult)
async def execute_plan(req: PlanExecutionRequest) -> PlanExecutionResult:
    """Debug / legacy: execute a fully materialized plan sequentially."""
    return await executor.execute_plan(req)


# ─────────────────────────────────────────────────────────────
# Bus RPC worker
# ─────────────────────────────────────────────────────────────

config = ChassisConfig(service_name=settings.service_name, service_version="0.1.0")


async def handle_exec_rpc(req: CortexExecStepRequestV1) -> dict:
    trace_id = req.trace_id

    logger.info("[CortexExec] recv trace_id=%s verb=%s step=%s calls=%d timeout_ms=%s", trace_id, req.verb_name, req.step_name, len(req.calls), req.timeout_ms)
    replies, elapsed_ms = await executor.execute_step_calls(
        trace_id=trace_id,
        verb_name=req.verb_name,
        step_name=req.step_name,
        order=req.order,
        origin_node=req.origin_node,
        calls=[c.model_dump(mode="json") for c in req.calls],
        timeout_ms=req.timeout_ms,
    )

    out = CortexExecStepReplyV1(
        trace_id=trace_id,
        ok=True,
        elapsed_ms=elapsed_ms,
        results=replies,
    )
    return out.model_dump(mode="json")


svc = Rabbit[
    CortexExecStepRequestV1,
    dict,
](
    config,
    intake_channel=settings.cortex_exec_request_channel,
    request_model=CortexExecStepRequestV1,
    handler=handle_exec_rpc,
    bus=bus,
)

_bg_task: asyncio.Task | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _bg_task
    _configure_logging()
    await bus.connect()
    _bg_task = asyncio.create_task(svc.run(), name="cortex-exec-rabbit")
    logger.info(
        "Cortex Exec online (http + bus) request_channel=%s",
        settings.cortex_exec_request_channel,
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _bg_task
    try:
        await svc.stop()
    finally:
        if _bg_task:
            _bg_task.cancel()
            await asyncio.gather(_bg_task, return_exceptions=True)
            _bg_task = None
        await bus.close()
