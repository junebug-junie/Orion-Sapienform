"""Deterministic guardrails after Mind LLM steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orion.mind.synthesis_v1 import (
    ActiveCognitiveFrontierV1,
    MindEvidencePackV1,
    MindStanceHandoffV1,
    SemanticClaimV1,
    SemanticSynthesisV1,
    SuppressedMindCandidateV1,
)
from orion.mind.validation import validate_merged_stance_brief_optional

from .evidence import (
    evidence_refs_in_pack,
    is_source_tag_label,
    normalize_evidence_refs_for_pack,
)

_MAX_CLAIMS = 24
_MAX_SELECTED = 8
_MAX_SAMPLE_FILTER_REASONS = 6
_MAX_SAMPLE_LABEL_LEN = 48


@dataclass
class SemanticClaimFilterStats:
    raw_claim_count: int = 0
    normalized_claim_count: int = 0
    schema_valid_claim_count: int = 0
    retained_claim_count: int = 0
    filtered_claim_count: int = 0
    filter_reasons_by_count: dict[str, int] = field(default_factory=dict)
    sample_filter_reasons: list[dict[str, str]] = field(default_factory=list)

    def dominant_filter_reason(self) -> str | None:
        if not self.filter_reasons_by_count:
            return None
        return max(self.filter_reasons_by_count.items(), key=lambda kv: kv[1])[0]

    def record_suppressed(self, *, label: str, reason: str, source_kind: str) -> None:
        self.filtered_claim_count += 1
        self.filter_reasons_by_count[reason] = self.filter_reasons_by_count.get(reason, 0) + 1
        if len(self.sample_filter_reasons) >= _MAX_SAMPLE_FILTER_REASONS:
            return
        short_label = (label or "(empty)").strip()
        if len(short_label) > _MAX_SAMPLE_LABEL_LEN:
            short_label = f"{short_label[: _MAX_SAMPLE_LABEL_LEN - 1].rstrip()}…"
        self.sample_filter_reasons.append(
            {"reason": reason, "label": short_label, "source_kind": source_kind}
        )


def filter_semantic_claims(
    synthesis: SemanticSynthesisV1,
    pack: MindEvidencePackV1,
    *,
    stats: SemanticClaimFilterStats | None = None,
) -> SemanticSynthesisV1:
    filtered, _ = filter_semantic_claims_with_stats(synthesis, pack, stats=stats)
    return filtered


def filter_semantic_claims_with_stats(
    synthesis: SemanticSynthesisV1,
    pack: MindEvidencePackV1,
    *,
    stats: SemanticClaimFilterStats | None = None,
) -> tuple[SemanticSynthesisV1, SemanticClaimFilterStats]:
    metrics = stats or SemanticClaimFilterStats()
    metrics.schema_valid_claim_count = len(synthesis.claims[:_MAX_CLAIMS])
    valid_refs = evidence_refs_in_pack(pack)
    kept: list[SemanticClaimV1] = []
    suppressed = list(synthesis.suppressed)
    for claim in synthesis.claims[:_MAX_CLAIMS]:
        label = (claim.label or "").strip()
        source_kind = claim.source_kinds[0] if claim.source_kinds else "unknown"
        if is_source_tag_label(label):
            suppressed.append(
                SuppressedMindCandidateV1(
                    label=label,
                    source_kind=source_kind,
                    reason="source_tag_not_semantic",
                )
            )
            metrics.record_suppressed(label=label, reason="source_tag_not_semantic", source_kind=source_kind)
            continue
        normalized_refs = normalize_evidence_refs_for_pack(list(claim.evidence_refs), pack)
        if normalized_refs != list(claim.evidence_refs):
            metrics.normalized_claim_count += 1
            claim = claim.model_copy(update={"evidence_refs": normalized_refs})
        if not normalized_refs or not all(ref in valid_refs for ref in normalized_refs):
            suppressed.append(
                SuppressedMindCandidateV1(
                    label=label or "(empty)",
                    source_kind=source_kind,
                    reason="unsupported_or_weak",
                )
            )
            metrics.record_suppressed(label=label, reason="unsupported_or_weak", source_kind=source_kind)
            continue
        if not label or not (claim.summary or "").strip():
            suppressed.append(
                SuppressedMindCandidateV1(
                    label=label or "(empty)",
                    source_kind=source_kind,
                    reason="empty",
                )
            )
            metrics.record_suppressed(label=label, reason="empty", source_kind=source_kind)
            continue
        kept.append(claim)
    metrics.retained_claim_count = len(kept)
    return synthesis.model_copy(update={"claims": kept, "suppressed": suppressed}), metrics


def evaluate_semantic_handoff_authorization(
    *,
    synthesis: SemanticSynthesisV1 | None,
    llm_errors: list[str],
) -> tuple[bool, bool, str, list[str]]:
    """Return (advisory_usable, stance_skip_authorized, authorization_reason, reasons)."""
    reasons: list[str] = []
    if llm_errors:
        reasons.append(f"llm_error:{llm_errors[0]}")
    if synthesis is None:
        reasons.append("semantic_synthesis_missing")
        return False, False, "semantic_synthesis_missing", reasons
    if not synthesis.claims:
        reasons.append("no_retained_semantic_claims")
        dominant = synthesis.diagnostics.notes[-1] if synthesis.diagnostics.notes else None
        if dominant:
            reasons.append(str(dominant))
        return False, False, "semantic_synthesis_empty", reasons
    if any(is_source_tag_label(c.label) for c in synthesis.claims):
        reasons.append("source_tag_claim_present")
        return False, False, "source_tag_claim_present", reasons
    return True, False, "semantic_claims_retained", ["evidence_backed_semantic_claims"]


def filter_active_frontier(
    frontier: ActiveCognitiveFrontierV1,
    pack: MindEvidencePackV1,
    *,
    claim_ids: set[str],
) -> ActiveCognitiveFrontierV1:
    valid_refs = evidence_refs_in_pack(pack)
    selected = []
    suppressed = list(frontier.suppressed)
    for matter in frontier.selected[:_MAX_SELECTED]:
        if is_source_tag_label(matter.label):
            suppressed.append(
                SuppressedMindCandidateV1(
                    label=matter.label,
                    source_kind="frontier",
                    reason="source_tag_not_semantic",
                )
            )
            continue
        if matter.source_claim_id not in claim_ids:
            suppressed.append(
                SuppressedMindCandidateV1(
                    label=matter.label,
                    source_kind="frontier",
                    reason="unsupported_or_weak",
                )
            )
            continue
        if matter.evidence_refs and not all(ref in valid_refs for ref in matter.evidence_refs):
            suppressed.append(
                SuppressedMindCandidateV1(
                    label=matter.label,
                    source_kind="frontier",
                    reason="evidence_too_weak",
                )
            )
            continue
        selected.append(matter)
    return frontier.model_copy(update={"selected": selected, "suppressed": suppressed})


def _stance_dominated_by_boilerplate(payload: dict[str, Any]) -> bool:
    joined = " ".join(
        str(payload.get(key) or "")
        for key in (
            "stance_summary",
            "self_relevance",
            "juniper_relevance",
            "user_intent",
        )
    ).lower()
    boilerplate_markers = (
        "contract_only",
        "contract-only",
        "no meaningful mind synthesis",
        "identity_yaml",
        "snapshot_source",
    )
    hits = sum(1 for marker in boilerplate_markers if marker in joined)
    return hits >= 2


def evaluate_stance_authorization(
    *,
    synthesis: SemanticSynthesisV1 | None,
    frontier: ActiveCognitiveFrontierV1 | None,
    stance_payload: dict[str, Any],
    llm_errors: list[str],
) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    if llm_errors:
        reasons.append(f"llm_error:{llm_errors[0]}")
    if synthesis is None or not synthesis.claims:
        reasons.append("no_evidence_backed_claims")
    if frontier is None or not frontier.selected:
        reasons.append("no_active_frontier_selected")
    valid, err = validate_merged_stance_brief_optional(stance_payload)
    if valid is None:
        reasons.append(f"stance_payload_invalid:{err or 'invalid'}")
    if synthesis and synthesis.claims:
        if any(is_source_tag_label(c.label) for c in synthesis.claims):
            reasons.append("source_tag_claim_present")
    if frontier and frontier.selected:
        if any(is_source_tag_label(m.label) for m in frontier.selected):
            reasons.append("source_tag_frontier_label")
    if _stance_dominated_by_boilerplate(stance_payload):
        reasons.append("contract_or_boilerplate_dominated")
    if reasons:
        return False, reasons, "invalid_handoff"
    return True, ["evidence_backed_claims", "active_frontier", "valid_chat_stance_brief"], "meaningful_synthesis"


def build_handoff(
    *,
    synthesis: SemanticSynthesisV1 | None,
    frontier: ActiveCognitiveFrontierV1 | None,
    stance_payload: dict[str, Any],
    model_id: str,
    llm_errors: list[str],
) -> MindStanceHandoffV1:
    authorized, reasons, quality = evaluate_stance_authorization(
        synthesis=synthesis,
        frontier=frontier,
        stance_payload=stance_payload,
        llm_errors=llm_errors,
    )
    hazards = list(frontier.hazards if frontier else [])
    return MindStanceHandoffV1(
        model_id=model_id,
        mind_quality=quality,  # type: ignore[arg-type]
        semantic_synthesis=synthesis,
        active_frontier=frontier,
        stance_payload=stance_payload,
        authorized_for_stance_use=authorized,
        authorization_reasons=reasons,
        hazards=hazards,
        diagnostics={"llm_errors": llm_errors},
    )
