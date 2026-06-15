"""Grounding preflight and answer-success evaluation for context-exec investigations."""

from __future__ import annotations

import re
from typing import Any

from orion.schemas.context_exec import ContextExecVerbStepV1

FAKE_ENGINES = frozenset({"fake", "mock", "smoke", "stub"})
RUNTIME_ONLY_CALLABLES = frozenset({"rlm_engine.run"})

_GROUNDING_PHRASES = (
    "do you recall",
    "do you remember",
    "where did",
    "what do you know",
    "look into",
    "tell me about",
    "what do you remember",
    "from memory",
    "in our history",
    "previously",
    "last time",
    "you said",
    "we discussed",
)

_EXPLICIT_FAKE_PHRASES = (
    "smoke test",
    "smoke run",
    "fake run",
    "fake engine",
    "stub run",
    "mock run",
)

_INVESTIGATION_COMPLETE_RE = re.compile(
    r"^investigation complete(?:\s+for:)?",
    re.IGNORECASE,
)


def grounding_required(
    text: str,
    *,
    mode: str,
    answer_contract: dict[str, Any] | None = None,
) -> bool:
    if answer_contract:
        if answer_contract.get("requires_runtime_grounding"):
            return True
        if answer_contract.get("requires_repo_grounding"):
            return True
    if mode != "general_investigation":
        return True
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _GROUNDING_PHRASES):
        return True
    if "?" in normalized:
        return True
    if any(token in normalized for token in ("recall", "remember", "memory", "trace", "file", "repo")):
        return True
    return False


def explicit_fake_run_requested(text: str, scopes: dict[str, Any] | None = None) -> bool:
    scope = scopes if isinstance(scopes, dict) else {}
    if scope.get("smoke_run") or scope.get("fake_engine") or scope.get("allow_fake_engine"):
        return True
    normalized = (text or "").strip().lower()
    return any(phrase in normalized for phrase in _EXPLICIT_FAKE_PHRASES)


def count_evidence(
    artifact: dict[str, Any],
    *,
    findings_bundle: Any = None,
    organ_cache: dict[str, Any] | None = None,
) -> int:
    total = 0
    for key in ("findings", "evidence", "supporting_evidence", "contradicting_evidence"):
        items = artifact.get(key)
        if isinstance(items, list):
            total += len(items)
    if findings_bundle is not None:
        findings = getattr(findings_bundle, "findings", None)
        if isinstance(findings, list) and findings:
            total = max(total, len(findings))
    cache = organ_cache if isinstance(organ_cache, dict) else {}
    recall = cache.get("recall")
    if isinstance(recall, dict):
        hits = recall.get("hits")
        if isinstance(hits, list) and hits:
            total = max(total, len(hits))
    traces = cache.get("traces")
    if isinstance(traces, list) and traces:
        total = max(total, len(traces))
    return total


def _grounding_sources_attempted(
    *,
    verb_trace: list[ContextExecVerbStepV1],
    runtime_debug: dict[str, Any],
    organ_cache: dict[str, Any] | None,
) -> bool:
    for step in verb_trace:
        callable_name = (step.callable or "").strip()
        if callable_name and callable_name not in RUNTIME_ONLY_CALLABLES:
            return True
    attempts = runtime_debug.get("grounding_attempts")
    if isinstance(attempts, dict) and any(attempts.values()):
        return True
    cache = organ_cache if isinstance(organ_cache, dict) else {}
    if cache.get("recall") is not None or cache.get("traces") is not None:
        return True
    return False


def _synthesis_status_label(*, model_synthesis_used: bool, runtime_debug: dict[str, Any]) -> str:
    if model_synthesis_used:
        return "completed"
    if runtime_debug.get("synthesis_fallback_used") or str(
        runtime_debug.get("synthesis_fallback_reason") or ""
    ).startswith("synthesis"):
        return "blocked"
    return "skipped"


def _blocked_summary(answer_status: str) -> str:
    if answer_status == "failed_fake_engine_selected":
        return (
            "Runtime completed, but this was not a grounded investigation. "
            "The selected engine was fake and no real recall or synthesis was performed."
        )
    if answer_status == "failed_grounding_preflight":
        return (
            "Runtime completed, but grounding failed: no recall, trace, or source evidence was captured."
        )
    return "No reliable grounded answer found."


def is_placeholder_investigation_summary(text: str) -> bool:
    return bool(_INVESTIGATION_COMPLETE_RE.match((text or "").strip()))


def evaluate_investigation_outcome(
    *,
    runtime_status: str,
    text: str,
    mode: str,
    artifact: dict[str, Any],
    runtime_debug: dict[str, Any],
    verb_trace: list[ContextExecVerbStepV1],
    findings_bundle: Any = None,
    organ_cache: dict[str, Any] | None = None,
    answer_contract: dict[str, Any] | None = None,
    scopes: dict[str, Any] | None = None,
    model_synthesis_used: bool = False,
    current_summary: str = "",
) -> dict[str, Any]:
    """Separate runtime success from grounded answer success."""
    runtime_ok = str(runtime_status or "").lower() == "ok"
    runtime_label = "ok" if runtime_ok else "failed"
    grounding_required_flag = grounding_required(text, mode=mode, answer_contract=answer_contract)
    evidence_count = count_evidence(artifact, findings_bundle=findings_bundle, organ_cache=organ_cache)
    grounding_attempted = _grounding_sources_attempted(
        verb_trace=verb_trace,
        runtime_debug=runtime_debug,
        organ_cache=organ_cache,
    )
    engine_selected = str(runtime_debug.get("engine_selected") or runtime_debug.get("engine") or "").lower()
    synthesis_status = _synthesis_status_label(
        model_synthesis_used=model_synthesis_used,
        runtime_debug=runtime_debug,
    )

    if not grounding_required_flag:
        return {
            "runtime_status": runtime_label,
            "answer_status": "answered_grounded" if runtime_ok else "failed",
            "grounding_status": "not_required",
            "synthesis_status": synthesis_status,
            "evidence_count": evidence_count,
            "grounding_required": False,
            "summary_text": current_summary,
        }

    if engine_selected in FAKE_ENGINES and not explicit_fake_run_requested(text, scopes):
        return {
            "runtime_status": runtime_label,
            "answer_status": "failed_fake_engine_selected",
            "grounding_status": "skipped",
            "synthesis_status": "skipped",
            "evidence_count": evidence_count,
            "grounding_required": True,
            "summary_text": _blocked_summary("failed_fake_engine_selected"),
        }

    if not grounding_attempted and evidence_count == 0:
        return {
            "runtime_status": runtime_label,
            "answer_status": "failed_grounding_preflight",
            "grounding_status": "skipped",
            "synthesis_status": synthesis_status,
            "evidence_count": 0,
            "grounding_required": True,
            "summary_text": _blocked_summary("failed_grounding_preflight"),
        }

    if evidence_count == 0 and not model_synthesis_used:
        summary = current_summary
        if is_placeholder_investigation_summary(summary):
            summary = _blocked_summary("failed_grounding_preflight")
        return {
            "runtime_status": runtime_label,
            "answer_status": "no_reliable_evidence",
            "grounding_status": "attempted",
            "synthesis_status": synthesis_status,
            "evidence_count": 0,
            "grounding_required": True,
            "summary_text": summary,
        }

    if evidence_count > 0 and grounding_attempted:
        if model_synthesis_used or mode == "general_investigation":
            answer_status = "answered_grounded"
        else:
            answer_status = "partial_or_weak_evidence"
        grounding_status = "attempted"
    elif model_synthesis_used:
        answer_status = "partial_or_weak_evidence"
        grounding_status = "attempted"
    else:
        answer_status = "no_reliable_evidence"
        grounding_status = "attempted"

    summary = current_summary
    if answer_status != "answered_grounded" and is_placeholder_investigation_summary(summary):
        summary = "No reliable grounded answer found."

    return {
        "runtime_status": runtime_label,
        "answer_status": answer_status,
        "grounding_status": grounding_status,
        "synthesis_status": synthesis_status,
        "evidence_count": evidence_count,
        "grounding_required": True,
        "summary_text": summary,
    }
