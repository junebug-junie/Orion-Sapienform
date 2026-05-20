"""Mind semantic synthesis step."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from pydantic import ValidationError

from orion.mind.synthesis_v1 import MindEvidencePackV1, SemanticSynthesisV1
from .guardrails import (
    SemanticClaimFilterStats,
    evaluate_semantic_handoff_authorization,
    filter_semantic_claims_with_stats,
)
from .llm_client import MindLLMClientProtocol
from .llm_context import MindLLMRequestContext
from .phase_telemetry import MindPhaseTelemetry

_SEMANTIC_SYSTEM = """You extract grounded semantic claims from evidence for an internal cognition organ.
You are NOT writing the user-facing reply.
Every claim MUST cite evidence_refs from evidence_refs_available in the user payload (exact strings only).
Labels must be semantic meanings, never source tags like identity_yaml, projection, autonomy, or social_bridge.
Identity background is not primary evidence unless the current turn explicitly involves identity, selfhood, role boundaries, or Orion/Juniper relationship framing.
Prefer uncertainty over invention.
Return strict JSON matching SemanticSynthesisV1 with schema_version mind.semantic_synthesis.v1.
Each claim object MUST include: claim_id, label, summary, claim_kind, evidence_refs, source_kinds.
Example claim:
{"claim_id":"c1","label":"shared evening plan","summary":"User mentions watching a show with Amanda.","claim_kind":"relationship_claim","evidence_refs":["current_turn:0"],"source_kinds":["current_turn"],"anchor":"relationship","confidence":0.85,"salience_hint":0.8,"recommended_effect":"receive_warmly"}
Do NOT emit legacy shapes like {"claim":"...","evidence":[...]} without the required fields."""


def _pack_prompt(pack: MindEvidencePackV1) -> str:
    items = [
        {
            "evidence_ref": it.evidence_ref,
            "source_kind": it.source_kind,
            "label": it.label,
            "text": it.text,
            "metadata": it.metadata,
        }
        for it in pack.items
    ]
    return json.dumps(
        {
            "current_user_text": pack.current_user_text,
            "evidence_refs_available": [it.evidence_ref for it in pack.items],
            "evidence_items": items,
        },
        ensure_ascii=False,
    )


def _short_label(text: str, *, max_len: int = 48) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 1].rstrip()}…"


def _legacy_claim_id(index: int, claim_text: str) -> str:
    digest = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()[:12]
    return f"legacy_{index}_{digest}"


def _has_current_claim_shape(item: dict[str, Any]) -> bool:
    return all(item.get(k) for k in ("claim_id", "label", "summary", "claim_kind"))


def _is_legacy_claim_shape(item: dict[str, Any]) -> bool:
    if _has_current_claim_shape(item):
        return False
    claim_text = item.get("claim")
    return isinstance(claim_text, str) and bool(claim_text.strip())


def _float_field(item: dict[str, Any], key: str, default: float) -> float:
    if key not in item:
        return default
    value = item.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _coerce_evidence_refs(item: dict[str, Any]) -> list[str]:
    refs = item.get("evidence_refs")
    if isinstance(refs, list):
        return [str(r).strip() for r in refs if str(r).strip()]
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        out: list[str] = []
        for entry in evidence:
            if isinstance(entry, str) and entry.strip():
                out.append(entry.strip())
            elif isinstance(entry, dict):
                ref = entry.get("evidence_ref") or entry.get("ref")
                if ref:
                    out.append(str(ref).strip())
        return [r for r in out if r]
    return []


def try_normalize_legacy_semantic_raw(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """Convert obvious legacy claim objects into SemanticSynthesisV1 claim shape."""
    claims_in = raw.get("claims")
    if not isinstance(claims_in, list) or not claims_in:
        return None, False

    normalized_claims: list[dict[str, Any]] = []
    saw_legacy = False
    for index, item in enumerate(claims_in):
        if not isinstance(item, dict):
            return None, False
        if _has_current_claim_shape(item):
            normalized_claims.append(item)
            continue
        if not _is_legacy_claim_shape(item):
            return None, False
        saw_legacy = True
        claim_text = str(item.get("claim") or "").strip()
        evidence_refs = _coerce_evidence_refs(item)
        source_kinds = item.get("source_kinds")
        if not isinstance(source_kinds, list) or not source_kinds:
            source_kinds = ["current_turn"] if evidence_refs else []
        default_kind = "current_turn_claim" if evidence_refs else "situation_claim"
        normalized_claims.append(
            {
                "claim_id": _legacy_claim_id(index, claim_text),
                "label": _short_label(claim_text),
                "summary": claim_text,
                "claim_kind": default_kind,
                "evidence_refs": evidence_refs,
                "source_kinds": [str(k) for k in source_kinds if str(k).strip()],
                "anchor": item.get("anchor") or "unknown",
                "confidence": _float_field(item, "confidence", 0.5),
                "salience_hint": _float_field(item, "salience_hint", 0.5),
                "recommended_effect": item.get("recommended_effect") or "no_effect",
                "metadata": {
                    **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
                    "normalized_from_legacy_claim_shape": True,
                },
            }
        )

    if not saw_legacy:
        return None, False

    out = dict(raw)
    out["schema_version"] = "mind.semantic_synthesis.v1"
    out["claims"] = normalized_claims
    diagnostics = dict(out.get("diagnostics") or {})
    notes = list(diagnostics.get("notes") or [])
    notes.append("normalized_from_legacy_claim_shape")
    diagnostics["notes"] = notes
    diagnostics["normalized_from_legacy_claim_shape"] = True
    out["diagnostics"] = diagnostics
    return out, True


def _validate_semantic_raw(raw: dict[str, Any]) -> tuple[SemanticSynthesisV1 | None, bool]:
    try:
        return SemanticSynthesisV1.model_validate(raw), False
    except ValidationError:
        normalized, did_normalize = try_normalize_legacy_semantic_raw(raw)
        if not did_normalize or normalized is None:
            return None, False
        try:
            return SemanticSynthesisV1.model_validate(normalized), True
        except ValidationError:
            return None, True


def run_semantic_synthesis(
    pack: MindEvidencePackV1,
    *,
    client: MindLLMClientProtocol,
    route: str,
    model_id: str,
    max_tokens: int,
    context: MindLLMRequestContext | None = None,
    timeout_sec: float | None = None,
) -> tuple[SemanticSynthesisV1 | None, str | None, MindPhaseTelemetry]:
    from .phase_telemetry import utc_now_iso

    started = utc_now_iso()
    t0 = time.perf_counter()
    raw, err, meta = client.request_json(
        system_prompt=_SEMANTIC_SYSTEM,
        user_prompt=_pack_prompt(pack),
        route=route,
        max_tokens=max_tokens,
        temperature=0.15,
        context=context,
        timeout_sec=timeout_sec,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = MindPhaseTelemetry(
        phase_name="semantic_synthesis",
        route=route,
        model=str(meta.get("model_used") or model_id),
        started_at=started,
        elapsed_ms=elapsed_ms,
        ok=False,
        parse_ok=raw is not None,
        error=err,
        token_usage=dict(meta.get("usage") or {}),
    )
    if raw is None:
        telemetry.status = "failed"
        return None, err or "semantic_synthesis_failed", telemetry
    if not isinstance(raw, dict):
        telemetry.validation_ok = False
        telemetry.status = "schema_invalid"
        telemetry.error = "semantic_schema_invalid:not_object"
        return None, "semantic_schema_invalid", telemetry

    synthesis, normalized_from_legacy = _validate_semantic_raw(raw)
    if synthesis is None:
        telemetry.validation_ok = False
        telemetry.ok = False
        telemetry.status = "schema_invalid"
        telemetry.error = "semantic_schema_invalid"
        return None, "semantic_schema_invalid", telemetry

    diag_notes = list(synthesis.diagnostics.notes)
    if normalized_from_legacy:
        diag_notes.append("normalized_from_legacy_claim_shape")
    synthesis = synthesis.model_copy(
        update={
            "model_id": model_id,
            "extraction_mode": "llm",
            "diagnostics": synthesis.diagnostics.model_copy(
                update={
                    "evidence_item_count": len(pack.items),
                    "projection_item_count_seen": int(pack.diagnostics.get("projection_item_count") or 0),
                    "recall_fragments_seen": int(pack.diagnostics.get("recall_fragment_count") or 0),
                    "autonomy_fields_seen": int(pack.diagnostics.get("autonomy_fields_seen") or 0),
                    "social_fields_seen": int(pack.diagnostics.get("social_fields_seen") or 0),
                    "llm_ok": True,
                    "llm_error": None,
                    "notes": diag_notes,
                }
            ),
        }
    )
    raw_claim_count = len(synthesis.claims)
    filter_stats = SemanticClaimFilterStats(raw_claim_count=raw_claim_count)
    filtered, filter_stats = filter_semantic_claims_with_stats(synthesis, pack, stats=filter_stats)
    advisory_ok, stance_skip_ok, auth_reason, auth_reasons = evaluate_semantic_handoff_authorization(
        synthesis=filtered,
        llm_errors=[],
    )
    telemetry.raw_claim_count = filter_stats.raw_claim_count
    telemetry.normalized_claim_count = filter_stats.normalized_claim_count
    telemetry.schema_valid_claim_count = filter_stats.schema_valid_claim_count
    telemetry.retained_claim_count = filter_stats.retained_claim_count
    telemetry.filtered_claim_count = filter_stats.filtered_claim_count
    telemetry.filter_reasons_by_count = dict(filter_stats.filter_reasons_by_count)
    telemetry.sample_filter_reasons = list(filter_stats.sample_filter_reasons)
    telemetry.authorization_reason = auth_reason
    telemetry.authorized_for_stance_use = advisory_ok
    telemetry.authorized_for_stance_skip = stance_skip_ok
    telemetry.validation_ok = bool(filtered.claims)
    if filtered.claims:
        telemetry.ok = True
        telemetry.status = "ok"
        if normalized_from_legacy:
            telemetry.error = None
    else:
        telemetry.ok = True
        telemetry.status = "filtered"
        dominant = filter_stats.dominant_filter_reason()
        telemetry.error = "semantic_synthesis_empty"
        diag_notes = list(filtered.diagnostics.notes)
        diag_notes.append(
            f"claims_generated={filter_stats.raw_claim_count};"
            f"claims_filtered={filter_stats.filtered_claim_count};"
            f"dominant_filter_reason={dominant or 'unknown'}"
        )
        if auth_reasons:
            diag_notes.append(f"authorization_failed:{';'.join(auth_reasons[:4])}")
        filtered = filtered.model_copy(
            update={
                "diagnostics": filtered.diagnostics.model_copy(update={"notes": diag_notes}),
            }
        )
    return filtered, None, telemetry
