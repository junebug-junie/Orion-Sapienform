"""Bridge Hub Agent mode to context-exec runs and UI payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from orion.schemas.context_exec import ContextExecPermissionV1, ContextExecRequestV1, ContextExecRunV1
from orion.schemas.cortex.contracts import AgentTraceStepV1, AgentTraceSummaryV1, CortexChatRequest

from scripts.context_exec_client import ContextExecClientError, agent_lane_enabled, run_context_exec


VALID_LLM_PROFILES = frozenset({"chat", "quick", "agent", "metacog"})


def normalize_llm_profile(raw: Any, *, default: str = "chat") -> str:
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
                effect_kind="read",
                status=item.status,
                summary=item.output_summary or item.input_summary or item.verb,
                duration_ms=item.duration_ms,
            )
        )
    return steps


def build_agent_trace_summary(
    *,
    run: ContextExecRunV1,
    correlation_id: Optional[str],
) -> Dict[str, Any]:
    steps = _steps_from_verb_trace(run)
    summary = AgentTraceSummaryV1(
        corr_id=correlation_id,
        mode="agent",
        status=run.status,
        step_count=len(steps),
        tool_call_count=len(steps),
        unique_tool_count=len({s.tool_id or s.event_type for s in steps}),
        summary_text=run.final_text or "",
        steps=steps,
        raw={
            "engine": "context_exec",
            "context_exec_mode": run.mode,
            "artifact_type": run.artifact_type,
            "runtime_debug": run.runtime_debug,
        },
    )
    return summary.model_dump(mode="json")


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
    routing["llm_profile"] = (run.runtime_debug or {}).get("llm_profile")
    return {
        "llm_response": run.final_text or "",
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
                },
                "engine": "context_exec",
            },
        },
        "mode": "agent",
        "correlation_id": correlation_id,
        "agent_trace": agent_trace,
        "routing_debug": routing,
        "context_exec_run": run.model_dump(mode="json"),
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
        or "chat"
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
