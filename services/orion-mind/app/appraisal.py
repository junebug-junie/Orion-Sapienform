"""Mind active cognitive frontier appraisal step."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SemanticSynthesisV1

from .evidence import MindEvidencePackV1
from .guardrails import filter_active_frontier
from .llm_client import MindLLMClientProtocol
from .llm_context import MindLLMRequestContext
from .phase_telemetry import MindPhaseTelemetry, utc_now_iso

_FRONTIER_SCHEMA_VERSION = "mind.active_cognitive_frontier.v1"
_SCHEMA_VERSION_ALIASES: dict[str, str] = {
    "active_cognitive_frontier.v1": _FRONTIER_SCHEMA_VERSION,
    "mind.active_cognitive_frontier.v1": _FRONTIER_SCHEMA_VERSION,
}
_SELECTED_LIST_KEYS = (
    "selected",
    "active_cognitive_frontier",
    "frontier",
    "matters",
    "selected_matters",
)
_VALID_MATTER_KINDS: frozenset[str] = frozenset(
    {
        "turn_anchor",
        "continuity_memory",
        "relationship_opportunity",
        "autonomy_pressure",
        "social_posture",
        "identity_boundary",
        "task_directive",
        "hazard",
        "curiosity_affordance",
        "uncertainty",
    }
)
_CLAIM_KIND_TO_MATTER_KIND: dict[str, str] = {
    "current_turn_claim": "turn_anchor",
    "continuity_claim": "continuity_memory",
    "relationship_claim": "relationship_opportunity",
    "task_claim": "task_directive",
    "identity_boundary_claim": "identity_boundary",
    "autonomy_claim": "autonomy_pressure",
    "social_claim": "social_posture",
    "situation_claim": "turn_anchor",
    "hazard_claim": "hazard",
    "curiosity_affordance_claim": "curiosity_affordance",
    "uncertainty_claim": "uncertainty",
}
_VALID_RECOMMENDED_EFFECTS: frozenset[str] = frozenset(
    {
        "answer_directly",
        "receive_warmly",
        "ask_one_situated_question",
        "suppress_question",
        "preserve_identity_boundary",
        "avoid_identity_sermon",
        "technical_triage",
        "surface_uncertainty",
        "no_effect",
    }
)

_APPRAISAL_SYSTEM = """You judge which semantic claims should shape the next assistant response.
Select a small active frontier (at most 6 matters). Defer or suppress the rest with reasons.
Preserve hazards. Do not let background identity dominate ordinary turns.
Include feature scores for selected matters.
Return strict JSON only (no prose) matching ActiveCognitiveFrontierV1 with schema_version mind.active_cognitive_frontier.v1.
Use key "selected" (not active_cognitive_frontier) for chosen matters.
Each selected item MUST include: matter_id, source_claim_id (from semantic claim_id), label, summary, matter_kind, score, features, evidence_refs, reason_selected, recommended_effect.
Example:
{"schema_version":"mind.active_cognitive_frontier.v1","selected":[{"matter_id":"m1","source_claim_id":"c1","label":"smoketest","summary":"User performed a smoketest.","matter_kind":"turn_anchor","score":0.85,"features":{"current_turn_relevance":0.9,"confidence":0.85},"evidence_refs":["current_turn:0"],"reason_selected":"grounded in current turn","recommended_effect":"answer_directly"}],"deferred":[],"suppressed":[],"hazards":[],"response_directives":[]}"""


def _float_field(item: dict[str, Any], key: str, default: float) -> float:
    value = item.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _resolve_source_claim_id(item: dict[str, Any], *, claim_ids: set[str]) -> str:
    for key in ("source_claim_id", "claim_id"):
        cid = str(item.get(key) or "").strip()
        if cid and (not claim_ids or cid in claim_ids):
            return cid
    if len(claim_ids) == 1:
        return next(iter(claim_ids))
    return str(item.get("source_claim_id") or item.get("claim_id") or "").strip()


def _coerce_matter_kind(item: dict[str, Any]) -> str:
    matter_kind = str(item.get("matter_kind") or "").strip()
    if matter_kind in _VALID_MATTER_KINDS:
        return matter_kind
    claim_kind = str(item.get("claim_kind") or "").strip()
    return _CLAIM_KIND_TO_MATTER_KIND.get(claim_kind, "turn_anchor")


def _coerce_recommended_effect(item: dict[str, Any]) -> str:
    effect = str(item.get("recommended_effect") or "").strip()
    if effect in _VALID_RECOMMENDED_EFFECTS:
        return effect
    return "no_effect"


def _coerce_features(item: dict[str, Any]) -> dict[str, Any]:
    features = item.get("features")
    if isinstance(features, dict) and features:
        return dict(features)
    confidence = _float_field(item, "confidence", 0.5)
    salience = _float_field(item, "salience_hint", 0.5)
    return {
        "current_turn_relevance": salience,
        "confidence": confidence,
        "evidence_strength": confidence,
    }


def _coerce_matter_item(
    item: dict[str, Any],
    *,
    index: int,
    claim_ids: set[str],
) -> dict[str, Any] | None:
    label = str(item.get("label") or "").strip()
    summary = str(item.get("summary") or "").strip()
    if not label or not summary:
        return None
    source_claim_id = _resolve_source_claim_id(item, claim_ids=claim_ids)
    if not source_claim_id:
        return None
    matter_id = str(item.get("matter_id") or "").strip() or f"m{index + 1}"
    return {
        "matter_id": matter_id,
        "source_claim_id": source_claim_id,
        "label": label,
        "summary": summary,
        "matter_kind": _coerce_matter_kind(item),
        "score": _float_field(item, "score", _float_field(item, "salience_hint", 0.5)),
        "features": _coerce_features(item),
        "evidence_refs": _coerce_string_list(item.get("evidence_refs")),
        "reason_selected": str(item.get("reason_selected") or "llm_selected").strip() or "llm_selected",
        "recommended_effect": _coerce_recommended_effect(item),
        "metadata": dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {},
    }


def _extract_selected_raw(raw: dict[str, Any]) -> tuple[list[Any] | None, str | None]:
    for key in _SELECTED_LIST_KEYS:
        value = raw.get(key)
        if isinstance(value, list) and value:
            return value, key
    return None, None


def _is_bare_frontier_matter_root(raw: dict[str, Any]) -> bool:
    selected_raw, _ = _extract_selected_raw(raw)
    if selected_raw:
        return False
    label = str(raw.get("label") or "").strip()
    summary = str(raw.get("summary") or "").strip()
    if not label or not summary:
        return False
    return bool(raw.get("matter_id") or raw.get("source_claim_id") or raw.get("claim_id"))


def try_normalize_frontier_raw(
    raw: dict[str, Any],
    *,
    claim_ids: set[str],
) -> tuple[dict[str, Any] | None, bool]:
    """Normalize common LLM frontier shapes (wrong schema_version, selected key aliases, claim-shaped matters)."""
    notes: list[str] = []
    selected_raw, selected_key = _extract_selected_raw(raw)
    if selected_raw is None and _is_bare_frontier_matter_root(raw):
        selected_raw = [raw]
        selected_key = "root_matter"
        notes.append("normalized_from_singleton_matter_root")

    if not selected_raw:
        return None, False

    if selected_key and selected_key != "selected":
        notes.append(f"normalized_selected_key:{selected_key}")

    schema_version = str(raw.get("schema_version") or "").strip()
    if schema_version in _SCHEMA_VERSION_ALIASES:
        if schema_version != _FRONTIER_SCHEMA_VERSION:
            notes.append("normalized_frontier_schema_version")
    elif not schema_version:
        notes.append("normalized_frontier_schema_version_missing")

    selected: list[dict[str, Any]] = []
    for index, item in enumerate(selected_raw[:6]):
        if not isinstance(item, dict):
            continue
        matter = _coerce_matter_item(item, index=index, claim_ids=claim_ids)
        if matter is not None:
            selected.append(matter)

    if not selected:
        return None, False

    deferred_in = raw.get("deferred")
    suppressed_in = raw.get("suppressed")
    out: dict[str, Any] = {
        "schema_version": _FRONTIER_SCHEMA_VERSION,
        "selected": selected,
        "deferred": list(deferred_in) if isinstance(deferred_in, list) else [],
        "suppressed": list(suppressed_in) if isinstance(suppressed_in, list) else [],
        "hazards": _coerce_string_list(raw.get("hazards")),
        "response_directives": _coerce_string_list(raw.get("response_directives")),
        "diagnostics": dict(raw.get("diagnostics") or {}) if isinstance(raw.get("diagnostics"), dict) else {},
    }
    diagnostics = dict(out["diagnostics"])
    diag_notes = list(diagnostics.get("notes") or [])
    diag_notes.extend(notes)
    diagnostics["notes"] = diag_notes
    out["diagnostics"] = diagnostics
    return out, True


def _validate_frontier_raw(
    raw: dict[str, Any],
    *,
    claim_ids: set[str],
) -> tuple[ActiveCognitiveFrontierV1 | None, bool]:
    normalized, did_normalize = try_normalize_frontier_raw(raw, claim_ids=claim_ids)
    candidate = normalized if did_normalize and normalized is not None else raw
    try:
        return ActiveCognitiveFrontierV1.model_validate(candidate), did_normalize
    except ValidationError:
        if did_normalize:
            return None, True
        retry, did_retry = try_normalize_frontier_raw(raw, claim_ids=claim_ids)
        if not did_retry or retry is None:
            return None, False
        try:
            return ActiveCognitiveFrontierV1.model_validate(retry), True
        except ValidationError:
            return None, True


def run_active_frontier_judge(
    synthesis: SemanticSynthesisV1,
    pack: MindEvidencePackV1,
    *,
    client: MindLLMClientProtocol,
    route: str,
    model_id: str,
    max_tokens: int,
    thinking: bool = False,
    context: MindLLMRequestContext | None = None,
    timeout_sec: float | None = None,
) -> tuple[ActiveCognitiveFrontierV1 | None, str | None, MindPhaseTelemetry]:
    user_prompt = json.dumps(
        {
            "current_user_text": pack.current_user_text,
            "semantic_claims": [c.model_dump(mode="json") for c in synthesis.claims],
            "evidence_refs_available": [it.evidence_ref for it in pack.items],
        },
        ensure_ascii=False,
    )
    started = utc_now_iso()
    t0 = time.perf_counter()
    raw, err, meta = client.request_json(
        system_prompt=_APPRAISAL_SYSTEM,
        user_prompt=user_prompt,
        route=route,
        max_tokens=max_tokens,
        temperature=0.1,
        thinking=thinking,
        context=context,
        timeout_sec=timeout_sec,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = MindPhaseTelemetry(
        phase_name="active_frontier_judge",
        route=route,
        model=str(meta.get("model_used") or model_id),
        started_at=started,
        elapsed_ms=elapsed_ms,
        ok=raw is not None and err is None,
        parse_ok=raw is not None,
        error=err,
        token_usage=dict(meta.get("usage") or {}),
    )
    if raw is None:
        telemetry.status = "failed"
        return None, err or "appraisal_failed", telemetry
    if not isinstance(raw, dict):
        telemetry.validation_ok = False
        telemetry.status = "schema_invalid"
        telemetry.error = "frontier_schema_invalid:not_object"
        return None, "frontier_schema_invalid", telemetry

    claim_ids = {c.claim_id for c in synthesis.claims}
    frontier, normalized_from_shape = _validate_frontier_raw(raw, claim_ids=claim_ids)
    if frontier is None:
        telemetry.validation_ok = False
        telemetry.status = "schema_invalid"
        telemetry.error = "frontier_schema_invalid"
        return None, "frontier_schema_invalid", telemetry

    diag_notes = list(frontier.diagnostics.notes)
    frontier = frontier.model_copy(
        update={
            "model_id": model_id,
            "diagnostics": frontier.diagnostics.model_copy(
                update={
                    "selected_count": len(frontier.selected),
                    "deferred_count": len(frontier.deferred),
                    "suppressed_count": len(frontier.suppressed),
                    "llm_ok": True,
                    "llm_error": None,
                    "notes": diag_notes,
                }
            ),
        }
    )
    filtered = filter_active_frontier(frontier, pack, claim_ids=claim_ids)
    telemetry.validation_ok = bool(filtered.selected)
    if filtered.selected:
        telemetry.ok = True
        telemetry.status = "ok"
        if normalized_from_shape:
            telemetry.error = None
    else:
        telemetry.ok = True
        telemetry.status = "filtered"
        telemetry.error = "frontier_selection_empty"
    return filtered, None, telemetry
