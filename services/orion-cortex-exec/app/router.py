# services/orion-cortex-exec/app/router.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef

from .executor import call_step_services, run_recall_step
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest, PlanExecutionResult, StepExecutionResult

logger = logging.getLogger("orion.cortex.router")


def _extract_final_text(steps: List[StepExecutionResult]) -> str:
    for step in reversed(steps):
        for payload in (step.result or {}).values():
            if not isinstance(payload, dict):
                continue
            text = payload.get("content") or payload.get("text")
            if text:
                return str(text)
            raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
            if raw and raw.get("text"):
                return str(raw.get("text"))
    return ""


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
        recall_debug: Dict[str, Any] = {}
        memory_used = False
        soft_failure = False

        extra = req.args.extra or {}
        mode = extra.get("mode") or ctx.get("mode") or "brain"
        recall_cfg = extra.get("recall") or ctx.get("recall") or {}
        recall_required = bool(recall_cfg.get("required", False))
        recall_enabled = bool(recall_cfg.get("enabled", True))

        ctx["verb"] = plan.verb_name
        needs_memory = recall_enabled or any(s.requires_memory for s in plan.steps)
        if needs_memory:
            recall_step, recall_debug, _ = await run_recall_step(
                bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
            )
            step_results.append(recall_step)
            memory_used = recall_step.status == "success"
            if recall_step.status != "success":
                overall_status = "fail" if recall_required else "partial"
                soft_failure = not recall_required
                if recall_required:
                    return PlanExecutionResult(
                        verb_name=plan.verb_name,
                        request_id=req.args.request_id,
                        status=overall_status,
                        blocked=False,
                        blocked_reason=None,
                        steps=step_results,
                        mode=mode,
                        final_text=None,
                        memory_used=memory_used,
                        recall_debug=recall_debug,
                        error=recall_step.error,
                    )

        for step in sorted(plan.steps, key=lambda s: s.order):
            step_res = await call_step_services(
                bus,
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=correlation_id,
            )
            step_results.append(step_res)

            if step_res.status != "success":
                overall_status = "partial" if len(step_results) > 1 else "fail"
                break

        final_text = _extract_final_text(step_results)
        if overall_status == "success" and soft_failure:
            overall_status = "partial"

        return PlanExecutionResult(
            verb_name=plan.verb_name,
            request_id=req.args.request_id,
            status=overall_status,
            blocked=False,
            blocked_reason=None,
            steps=step_results,
            mode=mode,
            final_text=final_text or None,
            memory_used=memory_used,
            recall_debug=recall_debug,
            error=None if overall_status == "success" else step_results[-1].error,
        )


# Backward-compat alias: earlier patches referenced PlanRouter.
PlanRouter = PlanRunner
