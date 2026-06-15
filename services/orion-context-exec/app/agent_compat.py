from __future__ import annotations

from typing import Any

from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult
from orion.schemas.context_exec import (
    ContextExecPermissionV1,
    ContextExecRequestV1,
    ContextExecRunV1,
    context_exec_permissions_for_llm_profile,
)

from .settings import settings


def agent_chain_request_to_context_exec(body: AgentChainRequest) -> ContextExecRequestV1:
    text = body.text or ""
    if settings.context_exec_investigation_v2_enabled:
        profile = str(body.response_profile or "agent").strip().lower()
        permissions = context_exec_permissions_for_llm_profile("agent")
        return ContextExecRequestV1(
            text=text,
            mode="investigation_v2",
            session_id=body.session_id,
            user_id=body.user_id,
            messages=list(body.messages or []),
            answer_contract=_parse_answer_contract(body.answer_contract),
            packs=list(body.packs or []),
            permissions=permissions,
            llm_profile=profile,
        )

    mode = "general_investigation"
    tl = text.lower()
    if "where did" in tl and ("claim" in tl or "belief" in tl or "come from" in tl):
        mode = "belief_provenance"
    elif any(t in tl for t in ["trace", "corr", "correlation", "run id", "fail open", "failed"]):
        mode = "trace_autopsy"
    elif "what breaks" in tl or "impact" in tl:
        mode = "repo_impact_analysis"
    permissions = ContextExecPermissionV1()
    if mode == "repo_impact_analysis":
        permissions = permissions.model_copy(update={"read_repo": True})
    return ContextExecRequestV1(
        text=text,
        mode=mode,  # type: ignore[arg-type]
        session_id=body.session_id,
        user_id=body.user_id,
        messages=list(body.messages or []),
        answer_contract=_parse_answer_contract(body.answer_contract),
        packs=list(body.packs or []),
        permissions=permissions,
        expected_artifact_type={
            "belief_provenance": "BeliefProvenanceReportV1",
            "trace_autopsy": "TraceAutopsyReportV1",
            "repo_impact_analysis": "RepoImpactAnalysisReportV1",
        }.get(mode),
    )


def _parse_answer_contract(raw: dict[str, Any] | None) -> AnswerContract | None:
    if not isinstance(raw, dict):
        return None
    try:
        return AnswerContract.model_validate(raw)
    except Exception:
        return None


def context_exec_run_to_agent_chain_result(run: ContextExecRunV1, *, mode: str = "agent") -> AgentChainResult:
    structured: dict[str, Any] = {
        "context_exec": {
            "run_id": run.run_id,
            "mode": run.mode,
            "artifact_type": run.artifact_type,
            "artifact": run.artifact,
            "findings_bundle": run.findings_bundle.model_dump(mode="json") if run.findings_bundle else None,
        },
    }
    if run.findings_bundle:
        structured["findings_bundle"] = run.findings_bundle.model_dump(mode="json")
    rd = dict(run.runtime_debug or {})
    rd.setdefault("engine", "context_exec")
    return AgentChainResult(
        mode=mode,
        text=run.final_text or run.text,
        structured=structured,
        planner_raw={"runtime_debug": rd, "trace": [s.model_dump(mode="json") for s in run.verb_trace]},
        runtime_debug=rd,
    )
