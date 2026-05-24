"""Pydantic models for the repair_pressure appraisal pipeline.

These models are NOT SubstrateMoleculeV1 — they live outside the substrate
grammar by design. Repair evidence and appraisal dimensions are domain
concepts that must not pollute the canonical substrate atoms/gradients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceKind = Literal[
    "specificity_demand",
    "trust_rupture",
    "coherence_gap",
    "repetition_failure",
    "operational_block",
    "explicit_repair_command",
    "assistant_accountability_demand",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RepairEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_molecule_id: str

    evidence_kind: EvidenceKind

    detector: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Audit only. Do not treat as machine contract.
    span: str | None = None
    features: dict[str, float] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=_utcnow)


class RepairPressureAppraisalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appraisal_id: str
    window_id: str
    appraisal_kind: Literal["repair_pressure"] = "repair_pressure"

    dimensions: dict[str, float]
    evidence: list[RepairEvidenceV1]
    causal_molecule_ids: list[str]

    confidence: float = Field(ge=0.0, le=1.0)
    summary: str | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
