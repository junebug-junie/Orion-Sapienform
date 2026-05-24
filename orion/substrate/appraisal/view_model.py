"""Presentation-only view model for the Substrate Effect UI.

Lives next to the appraiser so backend owns translation.  Frontend renders
this view as-is; it must not re-derive labels from raw appraisal/signal
objects.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orion.signals.models import OrionSignalV1

from .contract import REPAIR_PRESSURE_DEBUG_KEY
from .models import RepairEvidenceV1, RepairPressureAppraisalV1


# ── Label maps ──────────────────────────────────────────────────────────

KIND_LABELS: dict[str, str] = {
    "repair_pressure": "Repair pressure",
    "specificity_demand": "Specificity demand",
    "trust_rupture": "Trust rupture",
    "coherence_gap": "Coherence gap",
    "repetition_failure": "Repetition failure",
    "operational_block": "Operational block",
    "explicit_repair_command": "Explicit repair command",
    "assistant_accountability_demand": "Assistant accountability demand",
    "salience": "Substrate salience",
    "contradiction": "Substrate contradiction",
    "coherence": "Substrate coherence",
    "novelty": "Substrate novelty",
    "level": "Level",
    "confidence": "Confidence",
    "repair_concrete": "Repair concrete mode",
    "concrete_bias": "Concrete bias",
    "normal_chat": "Normal chat",
    "none": "None",
}


def pressure_label(value: float) -> str:
    if value >= 0.75:
        return "HIGH"
    if value >= 0.45:
        return "MEDIUM"
    if value >= 0.25:
        return "LOW"
    return "NONE"


def strength_label(value: float) -> str:
    if value >= 0.85:
        return "Very strong"
    if value >= 0.65:
        return "Strong"
    if value >= 0.45:
        return "Medium"
    if value >= 0.25:
        return "Low"
    return "Very low"


def confidence_label(value: float) -> str:
    if value >= 0.85:
        return "Very high"
    if value >= 0.65:
        return "High"
    if value >= 0.45:
        return "Medium"
    if value >= 0.25:
        return "Low"
    return "Very low"


# ── View-model schemas ──────────────────────────────────────────────────


class SubstrateOutcomeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appraisal_kind: str
    level: float
    level_label: str
    confidence: float
    confidence_label: str
    behavior_applied: str | None = None
    summary: str


class BehaviorDeltaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_before: str | None = None
    contract_after: str | None = None
    changed: bool
    rules_activated: list[str] = Field(default_factory=list)
    explanation: str | None = None


class CausalChainStepV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    title: str
    description: str
    detail: str | None = None
    linked_ids: list[str] = Field(default_factory=list)


class EvidenceCardV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_kind: str
    label: str
    strength_label: str
    score: float
    confidence: float
    source_span: str | None = None
    explanation: str
    meaning: str
    source_molecule_id: str | None = None


class ScorecardItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    label: str
    value: float
    value_label: str | None = None
    contribution: str | None = None


class ScorecardV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    items: list[ScorecardItemV1]
    final_label: str
    explanation: str | None = None


class MoleculeSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    molecule_id: str
    label: str
    explanation: str
    molecule_kind: str
    provenance_label: str | None = None


class SubstrateEffectViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_id: str
    message_id: str | None = None
    outcome: SubstrateOutcomeV1
    why: str | None = None
    behavior_delta: BehaviorDeltaV1 | None = None
    causal_chain: list[CausalChainStepV1] = Field(default_factory=list)
    evidence_cards: list[EvidenceCardV1] = Field(default_factory=list)
    scorecard: ScorecardV1 | None = None
    molecule_summaries: list[MoleculeSummaryV1] = Field(default_factory=list)
    raw_debug: dict[str, Any] | None = None


# ── Builder ────────────────────────────────────────────────────────────────

_EVIDENCE_MEANING: dict[str, str] = {
    "specificity_demand": (
        "The response should stop exploring and produce a usable implementation handoff."
    ),
    "trust_rupture": (
        "The user is signalling that prior assistant output was unreliable; "
        "the next response should acknowledge this and not repeat the failure mode."
    ),
    "coherence_gap": (
        "The response has been drifting or contradicting itself; "
        "the next response should converge to one explicit stance."
    ),
    "repetition_failure": (
        "The user is repeating a request the assistant has already received; "
        "the next response should address it directly instead of restating context."
    ),
    "operational_block": (
        "The user needs the answer to plug into another builder or pipeline; "
        "the answer must be structured enough to hand off directly."
    ),
    "explicit_repair_command": (
        "The user is constraining the response style explicitly; "
        "the next response must obey that constraint, not negotiate it."
    ),
    "assistant_accountability_demand": (
        "The user is holding the assistant accountable for prior turns; "
        "the next response should briefly acknowledge that, not deflect."
    ),
}


def _evidence_explanation(ev: RepairEvidenceV1) -> str:
    label = KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind)
    if ev.span:
        return f"Detected {label.lower()} from: \"{ev.span}\""
    return f"Detected {label.lower()} in the recent chat window."


def _evidence_card_label(ev: RepairEvidenceV1) -> str:
    return f"{KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind)} — {strength_label(ev.score)}"


def _modes_changed(
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
) -> tuple[str | None, str | None, bool]:
    """Return (before_str, after_str, changed) using a single normalization rule."""
    raw_before = contract_before.get("mode")
    raw_after = contract_after.get("mode")
    before_str = None if raw_before is None else str(raw_before)
    after_str = None if raw_after is None else str(raw_after)
    return before_str, after_str, before_str != after_str


def _build_outcome(
    appraisal: RepairPressureAppraisalV1 | None,
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
) -> SubstrateOutcomeV1:
    if appraisal is None:
        return SubstrateOutcomeV1(
            appraisal_kind="none",
            level=0.0,
            level_label="NONE",
            confidence=0.0,
            confidence_label="Very low",
            behavior_applied=None,
            summary="No substrate effect was recorded for this turn.",
        )
    level = float(appraisal.dimensions.get("level", 0.0))
    confidence = float(appraisal.confidence)
    before_mode, after_mode, changed = _modes_changed(contract_before, contract_after)
    behavior_applied = after_mode if changed else None
    lvl_lbl = pressure_label(level)
    if behavior_applied:
        summary = (
            f"Repair pressure was {lvl_lbl}, so Orion switched into "
            f"{KIND_LABELS.get(behavior_applied, behavior_applied)}."
        )
    else:
        summary = (
            f"Repair pressure was {lvl_lbl}. Orion did not change the response contract."
        )
    return SubstrateOutcomeV1(
        appraisal_kind="repair_pressure",
        level=level,
        level_label=lvl_lbl,
        confidence=confidence,
        confidence_label=confidence_label(confidence),
        behavior_applied=behavior_applied,
        summary=summary,
    )


def _build_behavior_delta(
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
    *,
    appraisal_present: bool = True,
) -> BehaviorDeltaV1:
    before_mode, after_mode, changed = _modes_changed(contract_before, contract_after)
    rules = list(contract_after.get("rules") or []) if changed else []
    if changed:
        explanation = (
            "The response contract was switched because repair pressure crossed "
            "the threshold defined by apply_repair_pressure_contract."
        )
    elif appraisal_present:
        explanation = (
            "No response contract change was applied because repair pressure was "
            "below threshold."
        )
    else:
        explanation = "No response contract change was applied."
    return BehaviorDeltaV1(
        contract_before=before_mode,
        contract_after=after_mode,
        changed=changed,
        rules_activated=rules,
        explanation=explanation,
    )


def _build_why(evidence: list[RepairEvidenceV1]) -> str | None:
    if not evidence:
        return None
    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)[:3]
    fragments: list[str] = []
    for ev in ranked:
        label = KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind).lower()
        if ev.span:
            fragments.append(f"{label} (\"{ev.span}\")")
        else:
            fragments.append(label)
    return "Detected: " + "; ".join(fragments) + "."


def _build_causal_chain(
    user_text: str,
    appraisal: RepairPressureAppraisalV1 | None,
    signal: OrionSignalV1 | None,
    evidence: list[RepairEvidenceV1],
    behavior_changed: bool,
    behavior_applied: str | None,
) -> list[CausalChainStepV1]:
    if appraisal is None:
        return []
    steps: list[CausalChainStepV1] = [
        CausalChainStepV1(
            index=1,
            title="Chat turn observed",
            description=(user_text[:160] + "…") if len(user_text) > 160 else user_text,
        ),
        CausalChainStepV1(
            index=2,
            title="Substrate created an observation molecule",
            description="This turn became shared substrate evidence.",
            linked_ids=list(appraisal.causal_molecule_ids),
        ),
    ]
    if evidence:
        top = sorted(evidence, key=lambda e: e.score, reverse=True)[:3]
        bullets = [
            f"{strength_label(e.score)} {KIND_LABELS.get(e.evidence_kind, e.evidence_kind).lower()}"
            for e in top
        ]
        steps.append(
            CausalChainStepV1(
                index=3,
                title="Repair evidence was detected",
                description="; ".join(bullets),
            )
        )
    steps.append(
        CausalChainStepV1(
            index=len(steps) + 1,
            title="Appraiser reduced the evidence",
            description=(
                f"repair_pressure level={appraisal.dimensions.get('level', 0.0):.2f}, "
                f"confidence={appraisal.confidence:.2f}"
            ),
            linked_ids=[appraisal.appraisal_id],
        )
    )
    if signal is not None:
        steps.append(
            CausalChainStepV1(
                index=len(steps) + 1,
                title="Signal emitted",
                description=f"{signal.organ_id} / {signal.signal_kind}",
                linked_ids=[signal.signal_id],
            )
        )
    if behavior_changed:
        steps.append(
            CausalChainStepV1(
                index=len(steps) + 1,
                title="Behavior changed",
                description=(
                    f"The response used {KIND_LABELS.get(behavior_applied, behavior_applied)}."
                    if behavior_applied
                    else "The response contract changed."
                ),
            )
        )
    else:
        steps.append(
            CausalChainStepV1(
                index=len(steps) + 1,
                title="Behavior unchanged",
                description="No response contract switch was applied.",
            )
        )
    return steps


def _build_evidence_cards(evidence: list[RepairEvidenceV1]) -> list[EvidenceCardV1]:
    cards: list[EvidenceCardV1] = []
    for ev in evidence:
        cards.append(
            EvidenceCardV1(
                evidence_kind=ev.evidence_kind,
                label=_evidence_card_label(ev),
                strength_label=strength_label(ev.score),
                score=float(ev.score),
                confidence=float(ev.confidence),
                source_span=ev.span,
                explanation=_evidence_explanation(ev),
                meaning=_EVIDENCE_MEANING.get(ev.evidence_kind, ""),
                source_molecule_id=ev.source_molecule_id,
            )
        )
    return cards


_SCORECARD_KEYS: tuple[str, ...] = (
    "specificity_demand",
    "operational_block",
    "explicit_repair_command",
    "trust_rupture",
    "coherence_gap",
    "repetition_failure",
    "assistant_accountability_demand",
    "salience",
    "contradiction",
    "coherence",
)


def _build_scorecard(appraisal: RepairPressureAppraisalV1 | None) -> ScorecardV1 | None:
    if appraisal is None:
        return None
    items: list[ScorecardItemV1] = []
    for key in _SCORECARD_KEYS:
        value = float(appraisal.dimensions.get(key, 0.0))
        items.append(
            ScorecardItemV1(
                key=key,
                label=KIND_LABELS.get(key, key),
                value=value,
                value_label=strength_label(value),
            )
        )
    items.sort(key=lambda item: item.value, reverse=True)
    level = float(appraisal.dimensions.get("level", 0.0))
    final = pressure_label(level)
    top_two = [item.label for item in items[:2] if item.value > 0.0]
    if len(top_two) == 1:
        explanation = (
            f"The score was {final.lower()} mostly because "
            f"{top_two[0]} was the strongest contributor."
        )
    elif top_two:
        explanation = (
            f"The score was {final.lower()} mostly because "
            + " and ".join(top_two)
            + " were the strongest contributors."
        )
    else:
        explanation = "No dimension exceeded zero."
    return ScorecardV1(
        title="Repair Pressure Scorecard",
        items=items,
        final_label=f"Repair pressure is {final}.",
        explanation=explanation,
    )


def _molecule_label(mol_id: str, evidence: list[RepairEvidenceV1]) -> str:
    if any(ev.source_molecule_id == mol_id for ev in evidence):
        return "Repair evidence"
    return "Chat observation"


def _molecule_explanation(mol_id: str, evidence: list[RepairEvidenceV1]) -> str:
    hits = [ev for ev in evidence if ev.source_molecule_id == mol_id]
    if not hits:
        return "Created from the current user turn."
    kinds = [KIND_LABELS.get(ev.evidence_kind, ev.evidence_kind) for ev in hits]
    spans = [f'"{ev.span}"' for ev in hits if ev.span]
    if spans:
        return f"Captured {', '.join(kinds).lower()} from {spans[0]}."
    return f"Captured {', '.join(kinds).lower()}."


def _build_molecule_summaries(
    appraisal: RepairPressureAppraisalV1 | None,
    evidence: list[RepairEvidenceV1],
) -> list[MoleculeSummaryV1]:
    if appraisal is None:
        return []
    seen: set[str] = set()
    summaries: list[MoleculeSummaryV1] = []
    for mol_id in appraisal.causal_molecule_ids:
        if mol_id in seen:
            continue
        seen.add(mol_id)
        summaries.append(
            MoleculeSummaryV1(
                molecule_id=mol_id,
                label=_molecule_label(mol_id, evidence),
                explanation=_molecule_explanation(mol_id, evidence),
                molecule_kind="observation",
                provenance_label="chat_turn",
            )
        )
    return summaries


def build_substrate_effect_view(
    *,
    turn_id: str,
    message_id: str | None,
    user_text: str,
    appraisal: RepairPressureAppraisalV1 | None,
    signal: OrionSignalV1 | None,
    evidence: list[RepairEvidenceV1],
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
    include_raw_debug: bool = True,
) -> SubstrateEffectViewV1:
    outcome = _build_outcome(appraisal, contract_before, contract_after)
    delta = _build_behavior_delta(
        contract_before,
        contract_after,
        appraisal_present=appraisal is not None,
    )
    chain = _build_causal_chain(
        user_text=user_text,
        appraisal=appraisal,
        signal=signal,
        evidence=evidence,
        behavior_changed=delta.changed,
        behavior_applied=outcome.behavior_applied,
    )
    raw_debug: dict[str, Any] | None = None
    if include_raw_debug:
        raw_debug = {
            "appraisal": appraisal.model_dump(mode="json") if appraisal else None,
            "signal": signal.model_dump(mode="json") if signal else None,
            "evidence": [ev.model_dump(mode="json") for ev in evidence],
            "contract_before": dict(contract_before),
            "contract_after": dict(contract_after),
        }
    return SubstrateEffectViewV1(
        turn_id=turn_id,
        message_id=message_id,
        outcome=outcome,
        why=_build_why(evidence),
        behavior_delta=delta,
        causal_chain=chain,
        evidence_cards=_build_evidence_cards(evidence),
        scorecard=_build_scorecard(appraisal),
        molecule_summaries=_build_molecule_summaries(appraisal, evidence),
        raw_debug=raw_debug,
    )
