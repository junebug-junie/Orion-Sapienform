"""Presentation-only view model for the Substrate Effect UI.

Lives next to the appraiser so backend owns translation.  Frontend renders
this view as-is; it must not re-derive labels from raw appraisal/signal
objects.
"""

from __future__ import annotations

from datetime import datetime
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
