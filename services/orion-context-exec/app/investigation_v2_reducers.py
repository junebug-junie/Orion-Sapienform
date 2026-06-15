"""Reducers: EvidenceBundle / SourceResult → InvestigationSectionV2 (PR3)."""

from __future__ import annotations

import re
from typing import Any

from orion.schemas.context_exec import (
    ContextExecRequestV1,
    EvidenceBundle,
    InvestigationReportV2,
    InvestigationSectionV2,
    InvestigationV2AnswerStatus,
    SourceResult,
    SourceStatus,
)

from .alexzhang_rlm_engine import _agent_chain_risk

_PATH_FROM_CLAIM = re.compile(r"^([^:]+):\d+")
_ENV_CONFIG_MARKERS = (".env", "settings.py", "docker-compose", "config", "requirements.txt")
_TEST_MARKERS = ("/tests/", "/test_", "tests/test_")


def _status_value(result: SourceResult | None) -> str:
    if result is None:
        return "skipped"
    return result.status.value


def _paths_from_findings(findings: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for item in findings:
        if not isinstance(item, dict):
            continue
        ref = item.get("source_ref")
        if isinstance(ref, str) and ref.startswith("repo:"):
            path = ref.removeprefix("repo:")
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        claim = str(item.get("claim") or "")
        match = _PATH_FROM_CLAIM.match(claim)
        if match:
            path = match.group(1)
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _filter_paths(paths: list[str], markers: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for path in paths:
        lowered = path.lower()
        if any(marker in lowered for marker in markers):
            out.append(path)
    return out


def _evidence_refs_from_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in findings:
        if not isinstance(item, dict):
            continue
        ref = item.get("source_ref")
        if ref:
            refs.append({"ref": ref, "claim": item.get("claim")})
    return refs[:20]


def reduce_repo_section(
    result: SourceResult | None,
    *,
    request_text: str,
) -> InvestigationSectionV2:
    if result is None:
        return InvestigationSectionV2(
            source="repo",
            status="skipped",
            title="Repository",
            summary="Repo probe not run.",
        )

    status = result.status.value
    if result.status == SourceStatus.blocked:
        return InvestigationSectionV2(
            source="repo",
            status=status,
            title="Repository",
            summary="Repo inspection blocked by permissions.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )
    if result.status == SourceStatus.unavailable:
        return InvestigationSectionV2(
            source="repo",
            status=status,
            title="Repository",
            summary=result.summary or "Repo source unavailable.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )
    if result.status == SourceStatus.error:
        return InvestigationSectionV2(
            source="repo",
            status=status,
            title="Repository",
            summary=result.summary or "Repo probe failed.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )
    if result.status == SourceStatus.no_hit:
        terms = result.metadata.get("search_terms") or []
        term_hint = f" ({len(terms)} search term(s))" if terms else ""
        return InvestigationSectionV2(
            source="repo",
            status=status,
            title="Repository",
            summary=f"Repo searched{term_hint}; no matching anchors found.",
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )

    paths = _paths_from_findings(result.findings)
    risk = _agent_chain_risk(paths)
    config_anchors = _filter_paths(paths, _ENV_CONFIG_MARKERS)
    test_paths = _filter_paths(paths, _TEST_MARKERS)
    breaking_surfaces: list[str] = []
    migration_notes: list[str] = []
    lowered_q = request_text.lower()
    if "context-exec" in lowered_q or "context exec" in lowered_q:
        breaking_surfaces.append("context-exec runner and organ wiring")
        migration_notes.append("verify CONTEXT_EXEC_ENABLED and agent lane flags")
    if "agent-chain" in lowered_q or "agent chain" in lowered_q:
        breaking_surfaces.append("cortex-exec AgentChainClient hop")
        migration_notes.append("feature flag CONTEXT_EXEC_ENABLED for agent lane cutover")

    summary_parts = [f"{len(paths)} affected path(s); risk={risk}."]
    if paths[:5]:
        names = [p.rsplit("/", 1)[-1] for p in paths[:5]]
        summary_parts.append(f"Files: {', '.join(names)}.")
    if config_anchors:
        summary_parts.append(f"Config/env anchors: {', '.join(p.rsplit('/', 1)[-1] for p in config_anchors[:4])}.")
    if test_paths:
        summary_parts.append(f"Tests likely affected: {', '.join(p.rsplit('/', 1)[-1] for p in test_paths[:4])}.")

    metadata = dict(result.metadata)
    metadata.update(
        {
            "affected_paths": paths[:20],
            "breaking_surfaces": breaking_surfaces,
            "config_anchors": config_anchors[:10],
            "tests_likely_affected": test_paths[:10],
            "migration_notes": migration_notes,
            "risk": risk,
        }
    )

    return InvestigationSectionV2(
        source="repo",
        status=status,
        title="Repository impact",
        summary=" ".join(summary_parts),
        findings=list(result.findings),
        evidence_refs=_evidence_refs_from_findings(result.findings),
        elapsed_ms=result.elapsed_ms,
        metadata=metadata,
    )


def reduce_traces_section(result: SourceResult | None) -> InvestigationSectionV2:
    if result is None:
        return InvestigationSectionV2(
            source="traces",
            status="skipped",
            title="Traces",
            summary="Trace probe not run.",
        )

    status = result.status.value
    if result.status == SourceStatus.blocked:
        return InvestigationSectionV2(
            source="traces",
            status=status,
            title="Traces",
            summary="Trace inspection blocked by permissions.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.unavailable:
        return InvestigationSectionV2(
            source="traces",
            status=status,
            title="Traces",
            summary=result.summary or "Trace source unavailable.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.error:
        return InvestigationSectionV2(
            source="traces",
            status=status,
            title="Traces",
            summary=result.summary or "Trace probe failed.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.no_hit:
        return InvestigationSectionV2(
            source="traces",
            status=status,
            title="Traces",
            summary="No matching trace or correlation evidence.",
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )

    hit_count = result.metadata.get("hit_count") or len(result.findings)
    corr_ids = [
        str(item.get("source_ref") or "")
        for item in result.findings
        if isinstance(item, dict) and item.get("source_ref")
    ][:5]
    summary = f"{hit_count} trace hit(s)."
    if corr_ids:
        summary += f" Handles: {', '.join(corr_ids)}."

    return InvestigationSectionV2(
        source="traces",
        status=status,
        title="Trace / correlation",
        summary=summary,
        findings=list(result.findings),
        evidence_refs=_evidence_refs_from_findings(result.findings),
        elapsed_ms=result.elapsed_ms,
        metadata=dict(result.metadata),
    )


def reduce_recall_section(result: SourceResult | None) -> InvestigationSectionV2:
    if result is None:
        return InvestigationSectionV2(
            source="recall",
            status="skipped",
            title="Recall",
            summary="Recall probe not run.",
        )

    status = result.status.value
    if result.status == SourceStatus.blocked:
        return InvestigationSectionV2(
            source="recall",
            status=status,
            title="Recall",
            summary="Recall inspection blocked by permissions.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.unavailable:
        err = result.error or result.summary or "recall dependency unavailable"
        return InvestigationSectionV2(
            source="recall",
            status=status,
            title="Recall",
            summary=f"Recall dependency failure: {err}",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )
    if result.status == SourceStatus.error:
        return InvestigationSectionV2(
            source="recall",
            status=status,
            title="Recall",
            summary=result.summary or "Recall probe error.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.no_hit:
        return InvestigationSectionV2(
            source="recall",
            status=status,
            title="Recall",
            summary="Recall searched; no matching hits.",
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )

    hit_count = result.metadata.get("hit_count") or len(result.findings)
    return InvestigationSectionV2(
        source="recall",
        status=status,
        title="Recall",
        summary=f"{hit_count} recall hit(s).",
        findings=list(result.findings),
        evidence_refs=_evidence_refs_from_findings(result.findings),
        elapsed_ms=result.elapsed_ms,
        metadata=dict(result.metadata),
    )


def reduce_memory_section(result: SourceResult | None) -> InvestigationSectionV2:
    if result is None:
        return InvestigationSectionV2(
            source="memory",
            status="skipped",
            title="Memory / provenance",
            summary="Memory probe not run.",
        )

    status = result.status.value
    if result.status == SourceStatus.blocked:
        return InvestigationSectionV2(
            source="memory",
            status=status,
            title="Memory / provenance",
            summary="Memory inspection blocked by permissions.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.unavailable:
        return InvestigationSectionV2(
            source="memory",
            status=status,
            title="Memory / provenance",
            summary=result.summary or "Memory source unavailable.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.no_hit:
        return InvestigationSectionV2(
            source="memory",
            status=status,
            title="Memory / provenance",
            summary="No namespace/local memory claim matches.",
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )

    hit_count = result.metadata.get("hit_count") or len(result.findings)
    return InvestigationSectionV2(
        source="memory",
        status=status,
        title="Memory / provenance",
        summary=f"{hit_count} memory claim hit(s).",
        findings=list(result.findings),
        evidence_refs=_evidence_refs_from_findings(result.findings),
        elapsed_ms=result.elapsed_ms,
        metadata=dict(result.metadata),
    )


def reduce_runtime_section(result: SourceResult | None) -> InvestigationSectionV2:
    if result is None:
        return InvestigationSectionV2(
            source="runtime",
            status="skipped",
            title="Runtime logs",
            summary="Runtime log probe not run.",
        )

    status = result.status.value
    if result.status == SourceStatus.blocked:
        return InvestigationSectionV2(
            source="runtime",
            status=status,
            title="Runtime logs",
            summary="Runtime log inspection blocked by permissions.",
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )
    if result.status == SourceStatus.skipped:
        return InvestigationSectionV2(
            source="runtime",
            status=status,
            title="Runtime logs",
            summary=result.summary or "Runtime log probe not wired yet.",
            elapsed_ms=result.elapsed_ms,
            metadata=dict(result.metadata),
        )

    return InvestigationSectionV2(
        source="runtime",
        status=status,
        title="Runtime logs",
        summary=result.summary or f"Runtime probe status={status}.",
        findings=list(result.findings),
        error=result.error,
        elapsed_ms=result.elapsed_ms,
        metadata=dict(result.metadata),
    )


def reduce_health_section(result: SourceResult | None) -> InvestigationSectionV2:
    if result is None:
        return InvestigationSectionV2(
            source="health",
            status="skipped",
            title="Health / dependencies",
            summary="Health probe not run.",
        )

    checks = result.metadata if isinstance(result.metadata, dict) else {}
    check_lines = [f"{k}={v}" for k, v in sorted(checks.items())]
    summary = result.summary or "Shallow dependency health snapshot."
    if check_lines:
        summary = f"{summary} ({'; '.join(check_lines)})"

    return InvestigationSectionV2(
        source="health",
        status=result.status.value,
        title="Health / dependencies",
        summary=summary,
        elapsed_ms=result.elapsed_ms,
        metadata=dict(result.metadata),
    )


def _bundle_entries(bundle: EvidenceBundle) -> list[tuple[str, SourceResult | None]]:
    return [
        ("repo", bundle.repo),
        ("traces", bundle.traces),
        ("recall", bundle.recall),
        ("memory", bundle.memory),
        ("runtime", bundle.runtime),
        ("health", bundle.health),
    ]


STRONG_HIT_SOURCES = frozenset({"repo", "traces", "memory"})
IMPORTANT_SOURCES = frozenset({"repo", "traces", "recall", "memory"})


def reduce_sections(
    bundle: EvidenceBundle,
    *,
    request_text: str,
) -> dict[str, InvestigationSectionV2]:
    return {
        "repo": reduce_repo_section(bundle.repo, request_text=request_text),
        "traces": reduce_traces_section(bundle.traces),
        "recall": reduce_recall_section(bundle.recall),
        "memory": reduce_memory_section(bundle.memory),
        "runtime": reduce_runtime_section(bundle.runtime),
        "health": reduce_health_section(bundle.health),
    }


def _compose_answer_status(
    bundle: EvidenceBundle,
    request: ContextExecRequestV1,
    sections: dict[str, InvestigationSectionV2],
) -> tuple[
    InvestigationV2AnswerStatus,
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    dict[str, str],
]:
    sources: dict[str, str] = {}
    hit_sources: list[str] = []
    unavailable_sources: list[str] = []
    blocked_sources: list[str] = []
    failed_sources: list[str] = []
    grounded_sources: list[str] = []
    ran_sources: list[str] = []

    for name, result in _bundle_entries(bundle):
        if result is None:
            continue
        status_key = result.status.value
        sources[name] = status_key
        if result.status == SourceStatus.hit:
            hit_sources.append(name)
            ran_sources.append(name)
            if name in STRONG_HIT_SOURCES:
                grounded_sources.append(name)
        elif result.status == SourceStatus.no_hit:
            ran_sources.append(name)
        elif result.status == SourceStatus.unavailable:
            unavailable_sources.append(name)
            ran_sources.append(name)
        elif result.status == SourceStatus.blocked:
            blocked_sources.append(name)
        elif result.status == SourceStatus.error:
            failed_sources.append(name)
            ran_sources.append(name)
        elif result.status == SourceStatus.skipped:
            ran_sources.append(name)

    perms = request.permissions
    allowed_important = [
        name
        for name, flag in (
            ("repo", perms.read_repo),
            ("traces", perms.read_redis_traces),
            ("recall", perms.read_recall),
            ("memory", perms.read_memory),
        )
        if flag
    ]

    if failed_sources and not hit_sources and not ran_sources:
        answer_status: InvestigationV2AnswerStatus = "error"
    elif blocked_sources and not ran_sources:
        answer_status = "blocked"
    elif grounded_sources and any(
        sections[s].findings or sections[s].metadata.get("affected_paths")
        for s in grounded_sources
        if s in sections
    ):
        if unavailable_sources or any(
            sources.get(s) == "no_hit" for s in allowed_important if s not in grounded_sources
        ):
            answer_status = "partial_grounding"
        else:
            answer_status = "answered_grounded"
    elif hit_sources:
        answer_status = "partial_grounding"
    elif unavailable_sources and not hit_sources:
        important_unavailable = [s for s in unavailable_sources if s in IMPORTANT_SOURCES]
        if important_unavailable:
            answer_status = "dependency_unavailable"
        else:
            answer_status = "no_reliable_evidence"
    elif ran_sources and not hit_sources:
        answer_status = "no_reliable_evidence"
    elif allowed_important and len(blocked_sources) >= len(allowed_important):
        answer_status = "blocked"
    else:
        answer_status = "no_reliable_evidence"

    return (
        answer_status,
        failed_sources,
        blocked_sources,
        unavailable_sources,
        grounded_sources,
        hit_sources,
        sources,
    )


def _deterministic_summary(
    *,
    answer_status: str,
    sections: dict[str, InvestigationSectionV2],
    grounded_sources: list[str],
    unavailable_sources: list[str],
    failed_sources: list[str],
    blocked_sources: list[str],
) -> str:
    parts: list[str] = []
    if grounded_sources:
        repo_sec = sections.get("repo")
        if repo_sec and repo_sec.status == "hit" and repo_sec.summary:
            parts.append(repo_sec.summary)
        else:
            parts.append(f"Grounded evidence from: {', '.join(grounded_sources)}.")
    elif any(sections[s].status == "hit" for s in sections if s in sections):
        hit_names = [s for s, sec in sections.items() if sec.status == "hit"]
        parts.append(f"Evidence from: {', '.join(hit_names)}.")
    if unavailable_sources:
        parts.append(f"Unavailable sources: {', '.join(unavailable_sources)}.")
    if failed_sources:
        parts.append(f"Failed sources: {', '.join(failed_sources)}.")
    if blocked_sources:
        parts.append(f"Blocked sources: {', '.join(blocked_sources)}.")
    if not parts:
        parts.append("No evidence sources produced hits.")
    parts.append(f"Answer status: {answer_status}.")
    return " ".join(parts)


def _limitations_and_next_actions(
    *,
    answer_status: str,
    unavailable_sources: list[str],
    failed_sources: list[str],
    blocked_sources: list[str],
    sections: dict[str, InvestigationSectionV2],
) -> tuple[list[str], list[str]]:
    limitations: list[str] = []
    next_actions: list[str] = []

    for src in unavailable_sources:
        sec = sections.get(src)
        if sec and sec.error:
            limitations.append(f"{src} unavailable: {sec.error}")
        else:
            limitations.append(f"{src} source unavailable.")

    for src in failed_sources:
        sec = sections.get(src)
        err = sec.error if sec else "probe error"
        limitations.append(f"{src} probe error: {err}")

    if blocked_sources:
        limitations.append(f"Permissions blocked: {', '.join(blocked_sources)}.")

    if answer_status == "no_reliable_evidence":
        next_actions.append("Refine the question or widen allowed evidence sources.")
    elif answer_status == "dependency_unavailable":
        next_actions.append("Restore unavailable dependencies (recall, traces, bus) and retry.")
    elif answer_status == "partial_grounding":
        if unavailable_sources:
            next_actions.append("Review unavailable sources; repo evidence may still be actionable.")
        repo_sec = sections.get("repo")
        if repo_sec and repo_sec.metadata.get("tests_likely_affected"):
            next_actions.append("Run affected tests/smokes before changing runtime.")
    elif answer_status == "answered_grounded":
        repo_sec = sections.get("repo")
        if repo_sec and repo_sec.metadata.get("migration_notes"):
            next_actions.extend(str(n) for n in repo_sec.metadata["migration_notes"][:3])

    return limitations, next_actions


def compose_investigation_report(
    bundle: EvidenceBundle,
    request: ContextExecRequestV1,
) -> InvestigationReportV2:
    sections = reduce_sections(bundle, request_text=request.text)
    (
        answer_status,
        failed_sources,
        blocked_sources,
        unavailable_sources,
        grounded_sources,
        _hit_sources,
        sources,
    ) = _compose_answer_status(bundle, request, sections)

    summary = _deterministic_summary(
        answer_status=answer_status,
        sections=sections,
        grounded_sources=grounded_sources,
        unavailable_sources=unavailable_sources,
        failed_sources=failed_sources,
        blocked_sources=blocked_sources,
    )
    limitations, next_actions = _limitations_and_next_actions(
        answer_status=answer_status,
        unavailable_sources=unavailable_sources,
        failed_sources=failed_sources,
        blocked_sources=blocked_sources,
        sections=sections,
    )

    raw_evidence = bundle.model_dump(mode="json")

    return InvestigationReportV2(
        answer_status=answer_status,
        summary=summary,
        sections=sections,
        sources=sources,
        failed_sources=failed_sources,
        blocked_sources=blocked_sources,
        unavailable_sources=unavailable_sources,
        grounded_sources=grounded_sources,
        limitations=limitations,
        next_actions=next_actions,
        evidence=bundle,
        raw_evidence=raw_evidence,
        text_received=request.text,
    )


def apply_synthesis_to_report(
    report: InvestigationReportV2,
    *,
    synthesis_summary: str | None,
    synthesis_failed: bool,
) -> InvestigationReportV2:
    limitations = list(report.limitations)
    summary = report.summary
    if synthesis_summary:
        summary = synthesis_summary
    elif synthesis_failed:
        msg = "LLM synthesis unavailable; deterministic evidence summary returned."
        if msg not in limitations:
            limitations.append(msg)
    return report.model_copy(update={"summary": summary, "limitations": limitations})
