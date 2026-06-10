from __future__ import annotations

from typing import Any

from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult
from orion.schemas.context_exec import ContextExecRequestV1, ContextExecRunV1


def agent_chain_request_to_context_exec(body: AgentChainRequest) -> ContextExecRequestV1:
    mode = "general_investigation"
    text = body.text or ""
    tl = text.lower()
    if "where did" in tl and ("claim" in tl or "belief" in tl or "come from" in tl):
        mode = "belief_provenance"
    elif any(t in tl for t in ["trace", "corr", "correlation", "run id", "fail open", "failed"]):
        mode = "trace_autopsy"
    elif "what breaks" in tl or "impact" in tl:
        mode = "repo_impact_analysis"
    ac = None
    if isinstance(body.answer_contract, dict):
        try:
            ac = AnswerContract.model_validate(body.answer_contract)
        except Exception:
            ac = None
    return ContextExecRequestV1(
        text=text,
        mode=mode,  # type: ignore[arg-type]
        session_id=body.session_id,
        user_id=body.user_id,
        messages=list(body.messages or []),
        answer_contract=ac,
        packs=list(body.packs or []),
        expected_artifact_type={
            "belief_provenance": "BeliefProvenanceReportV1",
            "trace_autopsy": "TraceAutopsyReportV1",
            "repo_impact_analysis": "RepoImpactAnalysisReportV1",
        }.get(mode),
    )


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
