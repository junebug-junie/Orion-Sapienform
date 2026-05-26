from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MotifObservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motif_id: str

    motif_kind: Literal[
        "field_pattern",
        "attention_pattern",
        "self_state_pattern",
        "proposal_policy_pattern",
        "dispatch_feedback_pattern",
        "absence_pattern",
        "stability_pattern",
    ]

    label: str

    recurrence_count: int = Field(ge=1)
    support_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    evidence_frame_ids: list[str] = Field(default_factory=list)
    dominant_dimensions: dict[str, float] = Field(default_factory=dict)
    dominant_channels: dict[str, float] = Field(default_factory=dict)

    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None

    reasons: list[str] = Field(default_factory=list)


class ExpectationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expectation_id: str

    trigger_motif_id: str

    expected_outcome_kind: Literal[
        "loaded_self_state",
        "attention_saturation",
        "policy_review_required",
        "read_only_approved",
        "dry_run_feedback",
        "absence_feedback",
        "reliability_clear",
        "execution_pressure_high",
        "resource_pressure_high",
        "contract_drift_persists",
        "transport_stable",
        "unknown",
    ]

    confidence_score: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(ge=1)

    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SparseTensorSliceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_id: str

    tensor_kind: Literal[
        "substrate_history_sparse",
        "motif_condition_outcome",
        "field_attention_self",
        "policy_dispatch_feedback",
    ]

    axes: list[str] = Field(default_factory=list)
    coordinates: list[dict[str, str]] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    evidence_refs: list[str] = Field(default_factory=list)


class SchemaCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_candidate_id: str

    candidate_kind: Literal[
        "concept_candidate",
        "habit_candidate",
        "prior_candidate",
        "ontology_candidate",
        "policy_candidate",
    ]

    label: str

    source_motif_ids: list[str] = Field(default_factory=list)
    source_expectation_ids: list[str] = Field(default_factory=list)

    support_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    proposed_schema: dict[str, object] = Field(default_factory=dict)

    promotion_status: Literal[
        "candidate_only",
        "needs_review",
        "rejected",
        "promoted_elsewhere",
    ] = "candidate_only"

    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class ConsolidationFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["consolidation.frame.v1"] = "consolidation.frame.v1"

    frame_id: str
    generated_at: datetime

    window_start: datetime
    window_end: datetime

    consolidation_policy_id: str = "consolidation_policy.v1"

    motif_observations: list[MotifObservationV1] = Field(default_factory=list)
    dominant_motifs: list[str] = Field(default_factory=list)

    expectations: list[ExpectationV1] = Field(default_factory=list)
    tensor_slices: list[SparseTensorSliceV1] = Field(default_factory=list)
    schema_candidates: list[SchemaCandidateV1] = Field(default_factory=list)

    source_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
