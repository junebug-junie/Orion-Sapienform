"""Pydantic models for the repair_pressure appraisal pipeline.

These models are NOT SubstrateMoleculeV1 — they live outside the substrate
grammar by design. Repair evidence and appraisal dimensions are domain
concepts that must not pollute the canonical substrate atoms/gradients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.repair_evidence import EvidenceKind, RepairEvidenceV1

__all__ = ["EvidenceKind", "RepairEvidenceV1", "RepairPressureAppraisalV1"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
