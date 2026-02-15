# services/orion-cortex-exec/app/router.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef

from .executor import call_step_services, run_recall_step
from .recall_utils import (
    has_inline_recall,
    recall_enabled_value,
    resolve_profile,
    should_run_recall,
)
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest, PlanExecutionResult, StepExecutionResult
from .supervisor import Supervisor
from .settings import settings
from orion.cognition.verb_activation import is_active

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
        recall_enabled = recall_enabled_value(recall_cfg)
        recall_required = bool(recall_cfg.get("required", False))
        verb_recall_profile = None
        if isinstance(plan.metadata, dict):
            verb_recall_profile = plan.metadata.get("recall_profile") or None
        ctx.setdefault("plan_recall_profile", verb_recall_profile)
        selected_profile, profile_source = resolve_profile(
            recall_cfg,
            verb_profile=verb_recall_profile,
        )

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
                "Recall selected profile=%s source=%s",
                selected_profile,
                profile_source,
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
        if mode == "brain" and plan.verb_name and not is_active(plan.verb_name, node_name=settings.node_name):
            logger.warning("Inactive verb blocked in router corr_id=%s verb=%s", correlation_id, plan.verb_name)
            return PlanExecutionResult(
                verb_name=plan.verb_name,
                request_id=req.args.request_id,
                status="fail",
                blocked=False,
                blocked_reason=None,
                steps=step_results,
                mode=mode,
                final_text=f"Verb '{plan.verb_name}' is inactive on node {settings.node_name}.",
                memory_used=memory_used,
                recall_debug=recall_debug,
                error=f"inactive_verb:{plan.verb_name}",
            )
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

        inline_recall = has_inline_recall(plan.steps)
        should_recall, recall_reason = should_run_recall(recall_cfg, plan.steps)

        if should_recall and not inline_recall:
            logger.info(
                "Recall resolved profile=%s source=%s gating=%s",
                selected_profile,
                profile_source,
                recall_reason,
            )
            recall_step, recall_debug, _ = await run_recall_step(
                bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                recall_profile=selected_profile,
                diagnostic=diagnostic,
            )
            step_results.append(recall_step)
            memory_used = recall_step.status == "success"
            recall_count = 0
            if isinstance(recall_step.result, dict):
                recall_payload = recall_step.result.get("RecallService")
                if isinstance(recall_payload, dict):
                    recall_count = int(recall_payload.get("count") or 0)
                    recall_debug = recall_payload
            if recall_required and recall_count == 0:
                if diagnostic:
                    logger.info(
                        "required recall empty; failing fast session_id=%s trace_id=%s",
                        ctx.get("session_id"),
                        ctx.get("trace_id"),
                    )
                return PlanExecutionResult(
                    verb_name=plan.verb_name,
                    request_id=req.args.request_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=mode,
                    final_text=None,
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error="recall_required_but_empty",
                )
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
            if inline_recall:
                recall_debug = {"skipped": "inline_recall_step_present"}
                logger.info(
                    "Recall skipped; inline RecallService step present",
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg, "diagnostic": diagnostic},
                )
            else:
                recall_debug = {"skipped": recall_reason}
                ctx["memory_used"] = False
                logger.info(
                    "Recall skipped by gating (%s)",
                    recall_reason,
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg, "diagnostic": diagnostic},
                )

        for step in sorted(plan.steps, key=lambda s: s.order):
            if step.verb_name and not is_active(step.verb_name, node_name=settings.node_name):
                logger.warning("Inactive step verb blocked corr_id=%s verb=%s", correlation_id, step.verb_name)
                step_results.append(
                    StepExecutionResult(
                        status="fail",
                        verb_name=step.verb_name,
                        step_name=step.step_name,
                        order=step.order,
                        result={"error": "inactive_verb", "verb": step.verb_name, "node": settings.node_name},
                        latency_ms=0,
                        node=settings.node_name,
                        logs=[f"reject <- inactive verb {step.verb_name}"],
                        error=f"inactive_verb:{step.verb_name}",
                    )
                )
                overall_status = "fail"
                break
            ctx["prior_step_results"] = [res.model_dump(mode="json") for res in step_results]
            step_res = await call_step_services(
                bus,
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=correlation_id,
                diagnostic=diagnostic,
            )
            step_results.append(step_res)
            if isinstance(step_res.result, dict) and "RecallService" in step_res.result:
                recall_debug = step_res.result.get("RecallService", {})
                memory_used = step_res.status == "success"
                ctx["memory_used"] = memory_used

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
