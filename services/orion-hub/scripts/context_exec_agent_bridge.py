"""Bridge Hub Agent mode to context-exec runs and UI payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from orion.schemas.context_exec import ContextExecPermissionV1, ContextExecRequestV1, ContextExecRunV1
from orion.schemas.cortex.contracts import AgentTraceStepV1, AgentTraceSummaryV1, AgentTraceToolStatV1, CortexChatRequest

from scripts.context_exec_client import ContextExecClientError, agent_lane_enabled, run_context_exec


VALID_LLM_PROFILES = frozenset({"chat", "quick", "agent", "metacog"})
RUNTIME_ONLY_CALLABLES = frozenset({"rlm_engine.run"})
FAILED_ANSWER_STATUSES = frozenset(
    {
        "failed_fake_engine_selected",
        "failed_grounding_preflight",
        "no_reliable_evidence",
        "failed",
    }
)


def normalize_llm_profile(raw: Any, *, default: str = "quick") -> str:
    route = str(raw or default).strip().lower()
    return route if route in VALID_LLM_PROFILES else default


def should_use_context_exec_agent_lane(req: CortexChatRequest) -> bool:
    return agent_lane_enabled() and str(req.mode or "").strip().lower() == "agent"


def _infer_context_exec_mode(text: str) -> str:
    tl = (text or "").lower()
    if "where did" in tl and ("claim" in tl or "belief" in tl or "come from" in tl):
        return "belief_provenance"
    if any(token in tl for token in ("trace", "corr", "correlation", "run id", "fail open", "failed")):
        return "trace_autopsy"
    if "what breaks" in tl or "impact" in tl or "repo" in tl:
        return "repo_impact_analysis"
    if "patch" in tl and "proposal" in tl:
        return "patch_proposal"
    if "memory" in tl and ("correction" in tl or "correct" in tl):
        return "memory_correction_proposal"
    return "general_investigation"


def build_context_exec_request(
    *,
    req: CortexChatRequest,
    prompt: str,
    llm_profile: str,
) -> ContextExecRequestV1:
    mode = _infer_context_exec_mode(prompt)
    permissions = ContextExecPermissionV1()
    if mode == "repo_impact_analysis":
        permissions = permissions.model_copy(update={"read_repo": True})

    return ContextExecRequestV1(
        text=prompt,
        mode=mode,  # type: ignore[arg-type]
        session_id=req.session_id,
        user_id=req.user_id,
        correlation_id=req.trace_id,
        messages=list(req.messages or []),
        packs=list(req.packs or []),
        permissions=permissions,
        expected_artifact_type={
            "belief_provenance": "BeliefProvenanceReportV1",
            "trace_autopsy": "TraceAutopsyReportV1",
            "repo_impact_analysis": "RepoImpactAnalysisReportV1",
            "patch_proposal": "ProposalEnvelopeV1",
            "memory_correction_proposal": "ProposalEnvelopeV1",
        }.get(mode),
        llm_profile=normalize_llm_profile(llm_profile),
    )


def _steps_from_verb_trace(run: ContextExecRunV1) -> List[AgentTraceStepV1]:
    steps: List[AgentTraceStepV1] = []
    for item in run.verb_trace or []:
        steps.append(
            AgentTraceStepV1(
                index=item.step_index,
                event_type=item.verb,
                tool_id=item.callable,
                tool_family="runtime",
                action_kind="inspect",
                effect_kind="read_only",
                status=item.status,
                summary=item.output_summary or item.input_summary or item.verb,
                duration_ms=item.duration_ms,
            )
        )
    return steps


def _user_visible_steps(steps: List[AgentTraceStepV1]) -> List[AgentTraceStepV1]:
    visible: List[AgentTraceStepV1] = []
    for step in steps:
        tool_id = (step.tool_id or "").strip()
        if tool_id in RUNTIME_ONLY_CALLABLES:
            continue
        visible.append(step)
    return visible


def _tool_stats_from_steps(steps: List[AgentTraceStepV1]) -> List[AgentTraceToolStatV1]:
    buckets: dict[str, AgentTraceToolStatV1] = {}
    for step in steps:
        tool_id = (step.tool_id or step.event_type or "unknown").strip() or "unknown"
        if tool_id not in buckets:
            buckets[tool_id] = AgentTraceToolStatV1(
                tool_id=tool_id,
                tool_family=step.tool_family or "unknown",
                action_kind=step.action_kind or "unknown",
                effect_kind=step.effect_kind or "unknown",
                count=0,
                duration_ms=0,
            )
        stat = buckets[tool_id]
        stat.count += 1
        stat.duration_ms = int(stat.duration_ms or 0) + int(step.duration_ms or 0)
    return list(buckets.values())


def _answer_evaluation(run: ContextExecRunV1) -> dict[str, Any]:
    dbg = run.runtime_debug or {}
    raw = dbg.get("answer_evaluation")
    return raw if isinstance(raw, dict) else {}


def _trace_status_for_run(run: ContextExecRunV1, answer_eval: dict[str, Any]) -> str:
    answer_status = str(answer_eval.get("answer_status") or "")
    if answer_status in FAILED_ANSWER_STATUSES:
        return answer_status
    return run.status


def build_agent_trace_summary(
    *,
    run: ContextExecRunV1,
    correlation_id: Optional[str],
) -> Dict[str, Any]:
    steps = _steps_from_verb_trace(run)
    tool_steps = _user_visible_steps(steps)
    tool_stats = _tool_stats_from_steps(tool_steps)
    answer_eval = _answer_evaluation(run)
    summary_text = str(answer_eval.get("summary_text") or run.final_text or "")
    summary = AgentTraceSummaryV1(
        corr_id=correlation_id,
        mode="agent",
        status=_trace_status_for_run(run, answer_eval),
        step_count=len(steps),
        tool_call_count=sum(tool.count for tool in tool_stats),
        unique_tool_count=len(tool_stats),
        summary_text=summary_text,
        tools=tool_stats,
        steps=steps,
        raw={
            "engine": "context_exec",
            "context_exec_mode": run.mode,
            "artifact_type": run.artifact_type,
            "runtime_debug": run.runtime_debug,
            "runtime_step_count": len(steps),
            "runtime_status": answer_eval.get("runtime_status") or run.status,
            "answer_status": answer_eval.get("answer_status"),
            "grounding_status": answer_eval.get("grounding_status"),
            "synthesis_status": answer_eval.get("synthesis_status"),
            "evidence_count": answer_eval.get("evidence_count"),
            "grounding_required": answer_eval.get("grounding_required"),
        },
    )
    return summary.model_dump(mode="json")


def _synthesis_status_label(runtime_debug: dict[str, Any]) -> str:
    if runtime_debug.get("model_synthesis_used"):
        return "used"
    if runtime_debug.get("synthesis_fallback_used") or str(
        runtime_debug.get("synthesis_fallback_reason") or ""
    ).startswith("synthesis"):
        return "fallback"
    return "skipped"


def format_agent_operator_inline(
    run: ContextExecRunV1,
    *,
    llm_profile: str | None = None,
) -> str:
    """Build Hub inline Agent operator response."""
    dbg = run.runtime_debug or {}
    answer_eval = _answer_evaluation(run)
    answer_status = str(answer_eval.get("answer_status") or "")
    op = run.operator_summary
    mode = (op.agent_mode if op else run.mode) or "unknown"
    route = (
        (op.route_used if op else None)
        or dbg.get("route_used")
        or llm_profile
        or dbg.get("llm_profile_selected")
        or "quick"
    )
    synthesis = _synthesis_status_label(dbg)
    result = (op.summary if op else None) or run.final_text or "No summary available."
    headline = (
        "Runtime completed (not a grounded investigation)"
        if answer_status in FAILED_ANSWER_STATUSES
        else "Agent run complete"
    )
    lines = [
        headline,
        f"Mode: {mode}",
        f"Route: {route}",
        f"Synthesis: {synthesis}",
        f"Result: {result}",
    ]
    if answer_status:
        lines.insert(1, f"Answer status: {answer_status}")
    proposal_id = (op.proposal_id if op else None) or dbg.get("proposal_id")
    proposal_status = (op.proposal_status if op else None) or dbg.get("ledger_status")
    if proposal_id:
        status_label = proposal_status or "pending_review"
        lines.append(f"Proposal: {proposal_id} {status_label}")
        lines.append("Open Pending Decisions to review.")
    lines.append("Mutation: none")
    return "\n".join(lines)


def build_context_exec_chat_response(
    *,
    run: ContextExecRunV1,
    correlation_id: str,
    route_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent_trace = build_agent_trace_summary(run=run, correlation_id=correlation_id)
    routing = dict(route_debug or {})
    routing["context_exec_lane"] = True
    routing["context_exec_mode"] = run.mode
    dbg = run.runtime_debug or {}
    routing["llm_profile"] = dbg.get("llm_profile_selected") or dbg.get("route_used")
    routing["route_used"] = dbg.get("route_used")
    routing["model_synthesis_used"] = dbg.get("model_synthesis_used")
    answer_eval = dbg.get("answer_evaluation") if isinstance(dbg.get("answer_evaluation"), dict) else {}
    if answer_eval:
        routing["answer_status"] = answer_eval.get("answer_status")
        routing["runtime_status"] = answer_eval.get("runtime_status")
        routing["grounding_status"] = answer_eval.get("grounding_status")
        routing["synthesis_status"] = answer_eval.get("synthesis_status")
        routing["evidence_count"] = answer_eval.get("evidence_count")
        routing["grounding_required"] = answer_eval.get("grounding_required")
    inline = format_agent_operator_inline(
        run,
        llm_profile=str(routing.get("llm_profile") or ""),
    )
    operator_summary = run.operator_summary.model_dump(mode="json") if run.operator_summary else None
    return {
        "llm_response": inline,
        "raw": {
            "ok": run.status == "ok",
            "mode": "agent",
            "status": run.status,
            "final_text": run.final_text,
            "metadata": {
                "context_exec": {
                    "run_id": run.run_id,
                    "mode": run.mode,
                    "artifact_type": run.artifact_type,
                    "artifact": run.artifact,
                    "operator_summary": operator_summary,
                },
                "engine": "context_exec",
            },
        },
        "mode": "agent",
        "correlation_id": correlation_id,
        "agent_trace": agent_trace,
        "routing_debug": routing,
        "context_exec_run": run.model_dump(mode="json"),
        "operator_summary": operator_summary,
    }


async def run_hub_agent_via_context_exec(
    *,
    req: CortexChatRequest,
    prompt: str,
    correlation_id: str,
    route_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    llm_profile = normalize_llm_profile(
        (req.options or {}).get("llm_route")
        or (route_debug or {}).get("llm_route")
        or "quick"
    )
    body = build_context_exec_request(req=req, prompt=prompt, llm_profile=llm_profile)
    try:
        run = await run_context_exec(body)
    except ContextExecClientError as exc:
        return {
            "error": str(exc),
            "error_code": "context_exec_unavailable",
            "mode": "agent",
            "correlation_id": correlation_id,
            "routing_debug": {
                **(route_debug or {}),
                "context_exec_lane": True,
                "llm_profile": llm_profile,
            },
        }
    return build_context_exec_chat_response(
        run=run,
        correlation_id=correlation_id,
        route_debug={
            **(route_debug or {}),
            "llm_profile": llm_profile,
            "context_exec_lane": True,
        },
    )
