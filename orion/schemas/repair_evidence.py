"""Repair pressure evidence schema (bus/registry layer).

Lives under orion.schemas so registry and pre_turn_appraisal imports do not
eagerly load orion.substrate.__init__ (SPARQL store stack → requests).
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
