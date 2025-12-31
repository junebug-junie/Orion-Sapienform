# services/orion-cortex-exec/app/router.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef

from .executor import call_step_services, run_recall_step
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest, PlanExecutionResult, StepExecutionResult
from .supervisor import Supervisor
from .settings import settings

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
        logger.info(
            "Running plan verb=%s mode=%s steps=%d correlation_id=%s",
            plan.verb_name,
            (req.args.extra or {}).get("mode") or ctx.get("mode") or "brain",
            len(plan.steps),
            correlation_id,
        )
        step_results: List[StepExecutionResult] = []
        overall_status = "success"
        recall_debug: Dict[str, Any] = {}
        memory_used = False
        soft_failure = False

        extra = req.args.extra or {}
        options = extra.get("options") if isinstance(extra, dict) else {}
        diagnostic = bool(
            settings.diagnostic_mode
            or extra.get("diagnostic")
            or (isinstance(options, dict) and (options.get("diagnostic") or options.get("diagnostic_mode")))
        )
        mode = extra.get("mode") or ctx.get("mode") or "brain"
        recall_cfg = extra.get("recall") or ctx.get("recall") or {}
        raw_enabled = recall_cfg.get("enabled", True)
        ctx.setdefault("recall", recall_cfg)
        recall_enabled = (
            str(raw_enabled).lower() not in {"false", "0", "no", "off"}
            if isinstance(raw_enabled, str)
            else bool(raw_enabled)
        )
        recall_required = bool(recall_cfg.get("required", False))

        if diagnostic:
            logger.info("Diagnostic PlanExecutionRequest json=%s", req.model_dump_json())
            logger.info(
                "Recall directive (raw) corr=%s enabled=%s required=%s cfg=%s",
                correlation_id,
                recall_enabled,
                recall_required,
                recall_cfg,
            )

        logger.info(
            "Exec plan start: corr=%s mode=%s verb=%s recall_enabled=%s recall_required=%s steps=%s recall_cfg=%s",
            correlation_id,
            mode,
            plan.verb_name,
            recall_enabled,
            recall_required,
            [s.step_name for s in plan.steps],
            recall_cfg,
        )

        options = extra.get("options") or ctx.get("options") or {}
        if isinstance(options, dict):
            ctx.setdefault("options", options)
            for key, val in options.items():
                ctx.setdefault(key, val)

        if diagnostic:
            ctx["diagnostic"] = True

        ctx["verb"] = plan.verb_name
        # Supervised path: delegate to Supervisor for agentic / council flows
        if mode in {"agent", "council"} or extra.get("supervised"):
            supervisor = Supervisor(bus)
            return await supervisor.execute(
                source=source,
                req=plan,
                correlation_id=correlation_id,
                ctx=ctx,
                recall_cfg=recall_cfg,
            )

        needs_memory = recall_enabled
        if needs_memory:
            recall_step, recall_debug, _ = await run_recall_step(
                bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                diagnostic=diagnostic,
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

        else:
            recall_debug = {"skipped": "disabled_by_client"}
            ctx["memory_used"] = False
            logger.info(
                "Recall skipped by client directive",
                extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg, "diagnostic": diagnostic},
            )

        for step in sorted(plan.steps, key=lambda s: s.order):
            step_res = await call_step_services(
                bus,
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=correlation_id,
                diagnostic=diagnostic,
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
