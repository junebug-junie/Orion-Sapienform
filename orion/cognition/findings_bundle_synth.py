"""Synthesize a serializable FindingsBundle from planner trace + contract (Phase 2 stub)."""

from __future__ import annotations

from typing import Any

from orion.schemas.cognition.answer_contract import Finding, FindingsBundle


def _trace_tool_ids(trace: list[Any]) -> list[str]:
    out: list[str] = []
    for step in trace or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action") or {}
        if isinstance(action, dict):
            tid = action.get("tool_id")
            if isinstance(tid, str) and tid.strip():
                out.append(tid.strip())
    return out


def _trace_has_repo_evidence(trace: list[Any]) -> bool:
    blob = str(trace).lower()
    hints = ("read_file", "file_read", "list_dir", "grep", "repo", "git ", "path:")
    return any(h in blob for h in hints)


def _trace_has_runtime_evidence(trace: list[Any]) -> bool:
    blob = str(trace).lower()
    hints = ("docker", "log", "journal", "kubectl", "systemd", "curl ", "http://", "https://")
    return any(h in blob for h in hints)


def synthesize_findings_bundle(
    *,
    answer_contract: dict[str, Any] | None,
    trace: list[Any],
    tools_called: list[str],
) -> dict[str, Any]:
    contract = answer_contract if isinstance(answer_contract, dict) else {}
    requires_repo = bool(contract.get("requires_repo_grounding"))
    requires_rt = bool(contract.get("requires_runtime_grounding"))
    findings: list[Finding] = []
    missing: list[str] = []

    if tools_called:
        findings.append(
            Finding(
                claim=f"Tools invoked: {', '.join(tools_called[-12:])}",
                evidence_type="inference",
                verified=False,
                confidence=0.35,
                scope="interpretation",
            )
        )

    if requires_repo and not _trace_has_repo_evidence(trace):
        missing.append("repo_file_reads_or_paths_in_trace")
    if requires_rt and not _trace_has_runtime_evidence(trace):
        missing.append("runtime_logs_or_commands_in_trace")

    if missing:
        grounded: Any = "insufficient_grounding"
    elif findings:
        grounded = "grounded_partial"
    else:
        grounded = "grounded_partial"

    bundle = FindingsBundle(
        findings=findings,
        missing_evidence=missing,
        unsupported_requests=[],
        next_checks=missing,
        grounded_status=grounded,
    )
    return bundle.model_dump(mode="json")


def merge_findings_bundle_dicts(base: dict[str, Any] | None, patch: dict[str, Any] | None) -> dict[str, Any]:
    """Merge two FindingsBundle-shaped dicts (exec supervisor aggregation)."""
    a = dict(base or {})
    b = dict(patch or {})
    merged_findings = list(a.get("findings") or []) + list(b.get("findings") or [])
    merged_missing = sorted(set(list(a.get("missing_evidence") or []) + list(b.get("missing_evidence") or [])))
    merged_unsup = sorted(set(list(a.get("unsupported_requests") or []) + list(b.get("unsupported_requests") or [])))
    merged_next = sorted(set(list(a.get("next_checks") or []) + list(b.get("next_checks") or [])))
    statuses = [str(a.get("grounded_status") or ""), str(b.get("grounded_status") or "")]
    grounded = "insufficient_grounding" if "insufficient_grounding" in statuses else "grounded_partial"
    return {
        "findings": merged_findings,
        "missing_evidence": merged_missing,
        "unsupported_requests": merged_unsup,
        "next_checks": merged_next,
        "grounded_status": grounded,
    }


def attach_findings_to_debug(
    dbg: dict[str, Any],
    *,
    answer_contract: dict[str, Any] | None,
    trace: list[Any],
    tools_called: list[str],
) -> None:
    dbg["findings_bundle"] = synthesize_findings_bundle(
        answer_contract=answer_contract,
        trace=trace,
        tools_called=tools_called,
    )
