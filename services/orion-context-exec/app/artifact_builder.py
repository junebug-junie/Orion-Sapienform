from __future__ import annotations

from typing import Any

from orion.schemas.cognition.answer_contract import Finding, FindingsBundle
from orion.schemas.context_exec import (
    BeliefProvenanceReportV1,
    ContextExecFindingV1,
    ContextExecMode,
    ContextExecRequestV1,
    ContextExecRunV1,
    RepoImpactAnalysisReportV1,
    TraceAutopsyReportV1,
)


def artifact_type_for_mode(mode: ContextExecMode) -> str | None:
    mapping = {
        "belief_provenance": "BeliefProvenanceReportV1",
        "trace_autopsy": "TraceAutopsyReportV1",
        "repo_impact_analysis": "RepoImpactAnalysisReportV1",
        "runtime_debug": "TraceAutopsyReportV1",
    }
    return mapping.get(mode)


def validate_artifact(mode: ContextExecMode, raw: Any) -> tuple[dict[str, Any], str | None, bool]:
    if not isinstance(raw, dict):
        return {}, None, False
    try:
        if mode == "belief_provenance":
            model = BeliefProvenanceReportV1.model_validate(raw)
        elif mode in {"trace_autopsy", "runtime_debug"}:
            model = TraceAutopsyReportV1.model_validate(raw)
        elif mode == "repo_impact_analysis":
            model = RepoImpactAnalysisReportV1.model_validate(raw)
        else:
            return raw, "GenericInvestigationV1", True
        return model.model_dump(mode="json"), model.__class__.__name__, True
    except Exception:
        return raw, artifact_type_for_mode(mode), False


def synthesize_findings_bundle(
    request: ContextExecRequestV1,
    artifact: dict[str, Any],
    *,
    schema_valid: bool,
) -> FindingsBundle | None:
    ac = request.answer_contract
    if ac is None and not (artifact.get("findings") or artifact.get("evidence")):
        return None
    raw_findings: list[Any] = list(artifact.get("findings") or artifact.get("evidence") or [])
    findings: list[Finding] = []
    verified_count = 0
    for item in raw_findings:
        if isinstance(item, ContextExecFindingV1):
            f = Finding(
                claim=item.claim,
                evidence_type=item.evidence_type,
                source_ref=item.source_ref,
                verified=item.verified,
                confidence=item.confidence,
                scope=item.scope,
            )
        elif isinstance(item, dict):
            f = Finding.model_validate(item)
        else:
            continue
        findings.append(f)
        if f.verified:
            verified_count += 1
    if not findings:
        status = "insufficient_grounding"
    elif verified_count == len(findings) and schema_valid:
        status = "grounded_complete"
    elif verified_count > 0:
        status = "grounded_partial"
    else:
        status = "insufficient_grounding"
    return FindingsBundle(
        findings=findings,
        grounded_status=status,
    )


def build_final_text(mode: ContextExecMode, artifact: dict[str, Any], *, status: str) -> str:
    if status != "ok":
        return f"Context-exec investigation could not complete ({status}). Insufficient grounding to answer with verified specifics."
    if mode == "belief_provenance":
        st = artifact.get("status", "unknown")
        claim = artifact.get("claim", "the claim")
        origin = artifact.get("likely_origin") or "unknown origin"
        return f"Belief provenance for '{claim}': status={st}. Likely origin: {origin}."
    if mode == "trace_autopsy":
        rc = artifact.get("root_cause") or "unknown"
        return f"Trace autopsy: {artifact.get('status', 'unknown')}. Root cause: {rc}."
    if mode == "repo_impact_analysis":
        return f"Repo impact: {artifact.get('status', 'unknown')}. Risk={artifact.get('risk', 'unknown')}."
    return str(artifact.get("summary") or artifact.get("target") or "Investigation complete.")
