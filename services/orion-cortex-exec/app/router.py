# services/orion-cortex-exec/app/router.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef

from .executor import call_step_services
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest, PlanExecutionResult, StepExecutionResult

logger = logging.getLogger("orion.cortex.router")


class PlanRunner:
    async def run_plan(
        self,
        bus: OrionBusAsync,
        *,
        source: ServiceRef,
        req: PlanExecutionRequest,
        correlation_id: str,
        ctx: Dict[str, Any],
    ) -> PlanExecutionResult:
        plan: ExecutionPlan = req.plan
        step_results: List[StepExecutionResult] = []
        overall_status = "success"

        for step in sorted(plan.steps, key=lambda s: s.order):
            step_res = await call_step_services(
                bus,
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=correlation_id,
            )
            step_results.append(step_res)

            if step_res.status != "success" and not step_res.soft_fail:
                overall_status = "partial" if len(step_results) > 1 else "fail"
                break
            if step_res.soft_fail and overall_status == "success":
                overall_status = "partial"

        return PlanExecutionResult(
            verb_name=plan.verb_name,
            request_id=req.args.request_id,
            status=overall_status,
            blocked=False,
            blocked_reason=None,
            steps=step_results,
        )


# Backward-compat alias: earlier patches referenced PlanRouter.
PlanRouter = PlanRunner
