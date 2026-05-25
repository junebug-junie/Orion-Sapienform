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

    source_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
