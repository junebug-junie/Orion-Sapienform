"""investigation_v2 evidence sweep runner (PR2)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from orion.schemas.context_exec import (
    ContextExecRequestV1,
    EvidenceBundle,
    InvestigationReportV2,
    InvestigationV2AnswerStatus,
    SourceResult,
    SourceStatus,
)

from .alexzhang_rlm_engine import _merge_repo_hits, _repo_search_terms
from .callable_namespace import ContextNamespace
from .organ_runtime import OrganRuntime
from .settings import settings

logger = logging.getLogger("orion-context-exec.investigation_v2")

INVESTIGATION_V2_ARTIFACT_TYPE = "InvestigationReportV2"
STRONG_HIT_SOURCES = frozenset({"repo", "traces", "memory"})


def _finding(
    *,
    claim: str,
    evidence_type: str,
    source_ref: str | None = None,
    verified: bool = False,
    confidence: float = 0.5,
    scope: str = "fact",
) -> dict[str, Any]:
    return {
        "claim": claim[:500],
        "evidence_type": evidence_type,
        "source_ref": source_ref,
        "verified": verified,
        "confidence": confidence,
        "scope": scope,
    }


def _trace_finding(hit: dict[str, Any]) -> dict[str, Any]:
    return _finding(
        claim=str(hit.get("snippet") or hit.get("handle") or "trace hit")[:200],
        evidence_type="runtime_log",
        source_ref=hit.get("handle"),
        verified=False,
        confidence=0.5,
    )


def _recall_finding(hit: dict[str, Any]) -> dict[str, Any]:
    return _finding(
        claim=str(hit.get("snippet") or hit.get("source_ref") or "recall hit")[:200],
        evidence_type="user_artifact",
        source_ref=hit.get("source_ref"),
        verified=False,
        confidence=float(hit.get("score") or 0.5),
    )


def _memory_finding(hit: dict[str, Any]) -> dict[str, Any]:
    return _finding(
        claim=str(hit.get("claim") or hit.get("snippet") or "memory hit")[:200],
        evidence_type="user_statement",
        source_ref=hit.get("source_ref"),
        verified=bool(hit.get("verified")),
        confidence=float(hit.get("confidence") or 0.5),
    )


def _probe_timeout_sec(request: ContextExecRequestV1) -> float:
    cap = float(settings.context_exec_investigation_v2_probe_timeout_sec)
    budget_cap = max(5.0, float(request.budget.max_seconds) / 6.0)
    return min(cap, budget_cap)


async def safe_probe(
    source: str,
    probe_fn: Callable[..., Awaitable[SourceResult]],
    *,
    permitted: bool,
    timeout_sec: float,
    **kwargs: Any,
) -> SourceResult:
    if not permitted:
        return SourceResult(
            source=source,
            status=SourceStatus.blocked,
            summary=f"{source} probe blocked by permissions",
        )
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(probe_fn(**kwargs), timeout=timeout_sec)
        elapsed = int((time.perf_counter() - started) * 1000)
        if result.elapsed_ms is None:
            result = result.model_copy(update={"elapsed_ms": elapsed})
        return result
    except TimeoutError:
        elapsed = int((time.perf_counter() - started) * 1000)
        return SourceResult(
            source=source,
            status=SourceStatus.unavailable,
            error=f"timeout after {timeout_sec}s",
            elapsed_ms=elapsed,
        )
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.warning("investigation_v2 probe %s failed: %s", source, exc)
        return SourceResult(
            source=source,
            status=SourceStatus.error,
            error=str(exc),
            elapsed_ms=elapsed,
        )


async def repo_probe(
    *,
    request: ContextExecRequestV1,
    namespace: ContextNamespace,
    runtime: OrganRuntime,
    organ_cache: dict[str, Any],
) -> SourceResult:
    if not settings.context_exec_real_repo_enabled:
        return SourceResult(
            source="repo",
            status=SourceStatus.unavailable,
            summary="repo reads disabled by settings",
        )
    terms = _repo_search_terms(request.text)
    hits: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for term in terms:
        if runtime is not None and request.permissions.read_repo:
            batch = runtime.repo_grep(term, limit=30)
        elif request.permissions.read_repo:
            batch = namespace.repo.grep(term, limit=30)
        else:
            batch = []
        _merge_repo_hits(hits, batch, seen_paths)

    organ_cache["repo_hits"] = hits
    organ_cache["repo_probe_attempted"] = True

    if not hits:
        return SourceResult(
            source="repo",
            status=SourceStatus.no_hit,
            summary=f"no repo matches for {len(terms)} search term(s)",
            metadata={"search_terms": terms},
        )

    findings = [
        _finding(
            claim=f"{h.get('path')}:{h.get('line_start')} {str(h.get('snippet', ''))[:120]}",
            evidence_type="repo_file",
            source_ref=h.get("source_ref"),
            verified=True,
            confidence=0.85,
        )
        for h in hits
        if isinstance(h, dict)
    ]
    return SourceResult(
        source="repo",
        status=SourceStatus.hit,
        summary=f"{len(hits)} repo hit(s) from {len(terms)} search term(s)",
        findings=findings,
        metadata={"search_terms": terms, "hit_count": len(hits)},
    )


async def trace_probe(
    *,
    request: ContextExecRequestV1,
    runtime: OrganRuntime,
    organ_cache: dict[str, Any],
) -> SourceResult:
    if not settings.context_exec_real_trace_enabled:
        return SourceResult(
            source="traces",
            status=SourceStatus.unavailable,
            summary="trace reads disabled by settings",
        )
    try:
        hits = await runtime.traces_search(
            query=request.text,
            limit=settings.context_exec_trace_limit,
        )
    except Exception as exc:
        return SourceResult(
            source="traces",
            status=SourceStatus.unavailable,
            error=str(exc),
        )

    organ_cache["traces"] = hits
    if not hits:
        return SourceResult(
            source="traces",
            status=SourceStatus.no_hit,
            summary="no matching traces",
        )
    return SourceResult(
        source="traces",
        status=SourceStatus.hit,
        summary=f"{len(hits)} trace hit(s)",
        findings=[_trace_finding(h) for h in hits if isinstance(h, dict)],
        metadata={"hit_count": len(hits)},
    )


async def recall_probe(
    *,
    request: ContextExecRequestV1,
    runtime: OrganRuntime,
    organ_cache: dict[str, Any],
) -> SourceResult:
    if not settings.context_exec_real_recall_enabled:
        return SourceResult(
            source="recall",
            status=SourceStatus.unavailable,
            summary="recall reads disabled by settings",
        )
    if settings.orion_bus_enabled and runtime.bus is None:
        return SourceResult(
            source="recall",
            status=SourceStatus.unavailable,
            summary="recall bus unavailable",
        )

    result = await runtime.recall_query(
        request.text,
        limit=settings.context_exec_recall_limit,
    )
    organ_cache["recall"] = result
    error = result.get("error") if isinstance(result, dict) else None
    if error:
        return SourceResult(
            source="recall",
            status=SourceStatus.unavailable,
            error=str(error),
            summary="recall RPC failed",
        )

    hits = result.get("hits") or [] if isinstance(result, dict) else []
    if not hits:
        return SourceResult(
            source="recall",
            status=SourceStatus.no_hit,
            summary="recall returned no hits",
        )
    return SourceResult(
        source="recall",
        status=SourceStatus.hit,
        summary=f"{len(hits)} recall hit(s)",
        findings=[_recall_finding(h) for h in hits if isinstance(h, dict)],
        metadata={"hit_count": len(hits)},
    )


async def memory_probe(
    *,
    request: ContextExecRequestV1,
    namespace: ContextNamespace,
    organ_cache: dict[str, Any],
) -> SourceResult:
    hits = namespace.memory.search_claims(request.text, limit=20)
    organ_cache["memory_hits"] = hits
    if not hits:
        return SourceResult(
            source="memory",
            status=SourceStatus.no_hit,
            summary="no memory claim matches",
        )
    return SourceResult(
        source="memory",
        status=SourceStatus.hit,
        summary=f"{len(hits)} memory hit(s)",
        findings=[_memory_finding(h) for h in hits if isinstance(h, dict)],
        metadata={"hit_count": len(hits)},
    )


async def runtime_probe(
    *,
    request: ContextExecRequestV1,
) -> SourceResult:
    if not request.permissions.read_runtime_logs:
        return SourceResult(
            source="runtime",
            status=SourceStatus.blocked,
            summary="runtime log probe blocked by permissions",
        )
    return SourceResult(
        source="runtime",
        status=SourceStatus.skipped,
        summary="runtime log probe not wired in PR2",
    )


async def health_probe(
    *,
    request: ContextExecRequestV1,
    runtime: OrganRuntime,
) -> SourceResult:
    checks: dict[str, str] = {
        "bus_enabled": "yes" if settings.orion_bus_enabled else "no",
        "bus_connected": "yes" if runtime.bus is not None else "no",
        "repo_enabled": "yes" if settings.context_exec_real_repo_enabled else "no",
        "recall_enabled": "yes" if settings.context_exec_real_recall_enabled else "no",
        "trace_enabled": "yes" if settings.context_exec_real_trace_enabled else "no",
    }
    return SourceResult(
        source="health",
        status=SourceStatus.no_hit,
        summary="shallow dependency health snapshot",
        metadata=checks,
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


def _compose_answer_status(
    bundle: EvidenceBundle,
    request: ContextExecRequestV1,
) -> tuple[InvestigationV2AnswerStatus, str, list[str], list[str], dict[str, str]]:
    sources: dict[str, str] = {}
    hit_sources: list[str] = []
    unavailable_sources: list[str] = []
    blocked_sources: list[str] = []
    failed_sources: list[str] = []
    ran_sources: list[str] = []

    for name, result in _bundle_entries(bundle):
        if result is None:
            continue
        status_key = result.status.value
        sources[name] = status_key
        if result.status == SourceStatus.hit:
            hit_sources.append(name)
            ran_sources.append(name)
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
    allowed_count = sum(
        1
        for flag in (
            perms.read_repo,
            perms.read_redis_traces,
            perms.read_recall,
            perms.read_memory,
            perms.read_runtime_logs,
        )
        if flag
    )
    allowed_count += 1  # health always attempted

    if blocked_sources and not ran_sources:
        answer_status: InvestigationV2AnswerStatus = "blocked"
    elif len(hit_sources) >= 2 and any(s in STRONG_HIT_SOURCES for s in hit_sources):
        answer_status = "answered_grounded"
    elif hit_sources:
        answer_status = "partial_grounding"
    elif unavailable_sources and not hit_sources:
        answer_status = "dependency_unavailable"
    elif ran_sources and not hit_sources:
        answer_status = "no_reliable_evidence"
    elif allowed_count == len(blocked_sources) and blocked_sources:
        answer_status = "blocked"
    else:
        answer_status = "no_reliable_evidence"

    parts: list[str] = []
    if hit_sources:
        parts.append(f"Evidence from: {', '.join(hit_sources)}")
    if unavailable_sources:
        parts.append(f"Unavailable: {', '.join(unavailable_sources)}")
    if failed_sources:
        parts.append(f"Errors: {', '.join(failed_sources)}")
    if blocked_sources:
        parts.append(f"Blocked: {', '.join(blocked_sources)}")
    if not parts:
        parts.append("No evidence sources produced hits")
    summary = "; ".join(parts)

    return answer_status, summary, failed_sources, blocked_sources, sources


def basic_investigation_v2_result(
    bundle: EvidenceBundle,
    request: ContextExecRequestV1,
) -> dict[str, Any]:
    answer_status, summary, failed_sources, blocked_sources, sources = _compose_answer_status(
        bundle,
        request,
    )
    report = InvestigationReportV2(
        answer_status=answer_status,
        summary=summary,
        sources=sources,
        failed_sources=failed_sources,
        blocked_sources=blocked_sources,
        evidence=bundle,
        text_received=request.text,
    )
    return report.model_dump(mode="json")


async def run_investigation_v2(
    request: ContextExecRequestV1,
    namespace: ContextNamespace,
    runtime: OrganRuntime,
    organ_cache: dict[str, Any],
) -> dict[str, Any]:
    bundle = EvidenceBundle()
    timeout_sec = _probe_timeout_sec(request)
    recall_timeout = min(timeout_sec, float(settings.context_exec_recall_timeout_sec))
    common = {
        "request": request,
        "namespace": namespace,
        "runtime": runtime,
        "organ_cache": organ_cache,
    }

    if request.permissions.read_repo:
        bundle.repo = await safe_probe(
            "repo",
            repo_probe,
            permitted=True,
            timeout_sec=timeout_sec,
            **common,
        )
    else:
        bundle.repo = await safe_probe("repo", repo_probe, permitted=False, timeout_sec=timeout_sec)

    if request.permissions.read_redis_traces:
        bundle.traces = await safe_probe(
            "traces",
            trace_probe,
            permitted=True,
            timeout_sec=timeout_sec,
            request=request,
            runtime=runtime,
            organ_cache=organ_cache,
        )
    else:
        bundle.traces = await safe_probe("traces", trace_probe, permitted=False, timeout_sec=timeout_sec)

    if request.permissions.read_recall:
        bundle.recall = await safe_probe(
            "recall",
            recall_probe,
            permitted=True,
            timeout_sec=recall_timeout,
            request=request,
            runtime=runtime,
            organ_cache=organ_cache,
        )
    else:
        bundle.recall = await safe_probe("recall", recall_probe, permitted=False, timeout_sec=recall_timeout)

    if request.permissions.read_memory:
        bundle.memory = await safe_probe(
            "memory",
            memory_probe,
            permitted=True,
            timeout_sec=timeout_sec,
            request=request,
            namespace=namespace,
            organ_cache=organ_cache,
        )
    else:
        bundle.memory = await safe_probe("memory", memory_probe, permitted=False, timeout_sec=timeout_sec)

    if request.permissions.read_runtime_logs:
        bundle.runtime = await safe_probe(
            "runtime",
            runtime_probe,
            permitted=True,
            timeout_sec=timeout_sec,
            request=request,
        )
    else:
        bundle.runtime = await safe_probe("runtime", runtime_probe, permitted=False, timeout_sec=timeout_sec)

    bundle.health = await safe_probe(
        "health",
        health_probe,
        permitted=True,
        timeout_sec=timeout_sec,
        request=request,
        runtime=runtime,
    )

    return basic_investigation_v2_result(bundle, request)
