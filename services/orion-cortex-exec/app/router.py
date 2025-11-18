# orion-cortex-exec/app/router.py

import logging
from typing import Dict, Any, List

from .models import (
    ExecutionPlan,
    PlanExecutionRequest,
    PlanExecutionResult,
    StepExecutionResult,
)
from .executor import StepExecutor

logger = logging.getLogger("orion-cortex.router")


class PlanRouter:
    """
    Plan-level orchestrator.

    - Accepts a full ExecutionPlan + args/context
    - Sequentially executes steps via StepExecutor
    - Aggregates results into PlanExecutionResult
    """

    def __init__(self, executor: StepExecutor | None = None):
        self.executor = executor or StepExecutor()

    async def execute_plan(
        self,
        req: PlanExecutionRequest,
    ) -> PlanExecutionResult:
        plan: ExecutionPlan = req.plan

        if plan.blocked:
            return PlanExecutionResult(
                verb_name=plan.verb_name,
                request_id=req.args.request_id,
                status="fail",
                blocked=True,
                blocked_reason=plan.blocked_reason,
                steps=[],
            )

        sorted_steps = sorted(plan.steps, key=lambda s: s.order)
        step_results: List[StepExecutionResult] = []

        overall_status = "success"

        for step in sorted_steps:
            step_res = await self.executor.execute_step(
                step=step,
                args=req.args.extra,
                context=req.context,
            )
            step_results.append(step_res)

            if step_res.status == "fail":
                overall_status = "partial" if step_results[:-1] else "fail"
                # For now, break on first failure. Later, we could allow best-effort.
                break

        return PlanExecutionResult(
            verb_name=plan.verb_name,
            request_id=req.args.request_id,
            status=overall_status,
            blocked=False,
            blocked_reason=None,
            steps=step_results,
        )
