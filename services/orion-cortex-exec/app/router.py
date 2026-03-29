# services/orion-cortex-exec/app/router.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef

from .executor import call_step_services, run_recall_step
from .recall_utils import (
    delivery_safe_recall_decision,
    has_inline_recall,
    recall_enabled_value,
    resolve_profile,
    should_run_recall,
)
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest, PlanExecutionResult, StepExecutionResult
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1
from .supervisor import Supervisor
from .settings import settings
from .metacog_enrichment import extract_reasoning_features
from orion.cognition.verb_activation import is_active

logger = logging.getLogger("orion.cortex.router")


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


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




def _normalize_execution_depth(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_metacog_traces(step_results: List[StepExecutionResult], *, correlation_id: str, session_id: str | None) -> List[MetacognitiveTraceV1]:
    traces: List[MetacognitiveTraceV1] = []
    for step in step_results:
        if not isinstance(step.result, dict):
            continue
        payload = step.result.get("LLMGatewayService")
        if not isinstance(payload, dict):
            continue
        reasoning_content = None
        reasoning_trace = None
        payload_keys = []
        if isinstance(payload, dict):
            payload_keys = sorted(payload.keys())
            reasoning_content = payload.get("reasoning_content")
            reasoning_trace = payload.get("reasoning_trace")
        else:
            try:
                dumped = payload.model_dump()
                payload_keys = sorted(dumped.keys())
                reasoning_content = dumped.get("reasoning_content")
                reasoning_trace = dumped.get("reasoning_trace")
            except Exception:
                payload_keys = [type(payload).__name__]
        trace_content = reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None
        print(
            "===THINK_HOP=== hop=exec_in "
            f"corr={correlation_id} "
            f"keys={payload_keys} "
            f"reasoning_len={len(reasoning_content) if isinstance(reasoning_content, str) else 0} "
            f"trace_len={len(trace_content) if isinstance(trace_content, str) else 0} "
            f"preview={_preview_text((reasoning_content if isinstance(reasoning_content, str) else None) or trace_content)}",
            flush=True,
        )
        reasoning = reasoning_content
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        model_name = payload.get("model_used") or payload.get("model") or "unknown"
        token_count = None
        usage = payload.get("usage")
        if isinstance(usage, dict):
            token_count = usage.get("completion_tokens") or usage.get("total_tokens")
        traces.append(
            MetacognitiveTraceV1(
                correlation_id=str(correlation_id),
                session_id=session_id,
                message_id=str(correlation_id),
                trace_role="reasoning" if step.step_name != "synthesize_chat_stance_brief" else "stance",
                trace_stage="post_answer",
                content=reasoning.strip(),
                model=str(model_name),
                token_count=int(token_count) if isinstance(token_count, (int, float)) else None,
                metadata={
                    "step_name": step.step_name,
                    "verb_name": step.verb_name,
                    **extract_reasoning_features(reasoning),
                },
            )
        )
        if _thought_debug_enabled():
            logger.info(
                "THOUGHT_DEBUG_EXEC stage=trace_collected corr=%s verb=%s step=%s trace_role=%s trace_stage=%s content_len=%s content_snippet=%r",
                correlation_id,
                step.verb_name,
                step.step_name,
                "reasoning" if step.step_name != "synthesize_chat_stance_brief" else "stance",
                "post_answer",
                _debug_len(reasoning),
                _debug_snippet(reasoning),
            )
    return traces


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
        depth = None
        if isinstance(plan.metadata, dict):
            depth = _normalize_execution_depth(plan.metadata.get("execution_depth"))
        start_mode = (req.args.extra or {}).get("mode") or ctx.get("mode") or "brain"
        logger.info(
            "plan_start corr_id=%s depth=%s mode=%s verb=%s steps=%s",
            correlation_id,
            depth,
            start_mode,
            plan.verb_name,
            [s.step_name for s in plan.steps],
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
        recall_policy = delivery_safe_recall_decision(
            recall_cfg,
            plan.steps,
            output_mode=ctx.get("output_mode"),
            verb_profile=verb_recall_profile,
            user_text=(ctx.get("raw_user_text") or ""),
            runtime_mode=mode,
        )
        selected_profile = recall_policy["profile"]
        profile_source = recall_policy["profile_source"]
        ctx.setdefault("debug", {})["recall_gating_reason"] = recall_policy["recall_gating_reason"]
        ctx.setdefault("debug", {})["recall_profile"] = selected_profile
        ctx.setdefault("debug", {})["recall_profile_source"] = profile_source
        ctx.setdefault("debug", {})["recall_profile_override_source"] = recall_policy.get("profile_override_source")

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
                "Recall selected profile=%s source=%s override_source=%s gating_reason=%s",
                selected_profile,
                profile_source,
                recall_policy.get("profile_override_source"),
                recall_policy["recall_gating_reason"],
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
        existing_scope = str(ctx.get("_run_scope_corr_id") or "")
        if existing_scope and existing_scope != correlation_id:
            logger.warning(
                "router_scope_reset corr_id=%s previous_scope=%s",
                correlation_id,
                existing_scope,
            )
        ctx["_run_scope_corr_id"] = correlation_id
        ctx["prior_step_results"] = []
        ctx.setdefault("prior_step_results_by_corr", {})[correlation_id] = []
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
        # Supervised path only for depth=2 agent runtime flows
        if (depth == 2) or (mode == "agent" and plan.verb_name == "agent_runtime") or extra.get("supervised"):
            supervisor = Supervisor(bus)
            return await supervisor.execute(
                source=source,
                req=plan,
                correlation_id=correlation_id,
                ctx=ctx,
                recall_cfg=recall_cfg,
            )

        inline_recall = has_inline_recall(plan.steps)
        should_recall = bool(recall_policy["run_recall"])
        recall_reason = str(recall_policy["reason"])

        if should_recall and not inline_recall:
            logger.info(
                "Recall resolved profile=%s source=%s gating=%s recall_gating_reason=%s",
                selected_profile,
                profile_source,
                recall_reason,
                recall_policy["recall_gating_reason"],
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
            if isinstance(recall_debug, dict):
                recall_debug.setdefault("profile", selected_profile)
                recall_debug.setdefault("profile_source", profile_source)
                recall_debug.setdefault("profile_override_source", recall_policy.get("profile_override_source"))
                recall_debug.setdefault("recall_gating_reason", recall_policy.get("recall_gating_reason"))
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
                recall_debug = {
                    "skipped": "inline_recall_step_present",
                    "recall_gating_reason": recall_policy["recall_gating_reason"],
                }
                logger.info(
                    "Recall skipped; inline RecallService step present",
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg, "diagnostic": diagnostic},
                )
            else:
                recall_debug = {
                    "skipped": recall_reason,
                    "recall_gating_reason": recall_policy["recall_gating_reason"],
                }
                ctx["memory_used"] = False
                logger.info(
                    "Recall skipped by gating (%s) recall_gating_reason=%s",
                    recall_reason,
                    recall_policy["recall_gating_reason"],
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
            scoped = ctx.setdefault("prior_step_results_by_corr", {})
            if isinstance(scoped, dict):
                scoped[correlation_id] = list(ctx["prior_step_results"])
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

        if depth == 1:
            logger.info("depth1_complete corr_id=%s verb=%s elapsed=%s", correlation_id, plan.verb_name, sum([s.latency_ms for s in step_results]))
        metacog_traces = _collect_metacog_traces(
            step_results,
            correlation_id=correlation_id,
            session_id=str(ctx.get("session_id")) if ctx.get("session_id") else None,
        )
        logger.info(
            "cortex_exec_metacog_attached corr_id=%s verb=%s traces=%s",
            correlation_id,
            plan.verb_name,
            len(metacog_traces),
        )
        if _thought_debug_enabled():
            logger.info(
                "THOUGHT_DEBUG_EXEC stage=metacog_attached corr=%s verb=%s traces=%s final_text_len=%s fallback_to_final_text=%s",
                correlation_id,
                plan.verb_name,
                len(metacog_traces),
                _debug_len(final_text),
                len(metacog_traces) == 0,
            )

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
            metacog_traces=metacog_traces,
            error=None if overall_status == "success" else step_results[-1].error,
        )


# Backward-compat alias: earlier patches referenced PlanRouter.
PlanRouter = PlanRunner
