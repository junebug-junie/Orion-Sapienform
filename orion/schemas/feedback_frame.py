from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OutcomeObservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str

    source_kind: Literal[
        "dispatch_candidate",
        "policy_decision",
        "proposal_candidate",
        "cortex_result",
        "field_delta",
        "attention_delta",
        "self_state_delta",
        "absence",
        "operator_feedback",
    ]

    source_id: str

    outcome_kind: Literal[
        "not_attempted",
        "dry_run",
        "prepared",
        "prepared_for_dispatch",
        "dispatched",
        "completed",
        "failed",
        "blocked",
        "deferred",
        "absent",
        "stale",
        "improved",
        "worsened",
        "unchanged",
        "unknown",
    ]

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    observed_at: datetime


class FeedbackFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["feedback.frame.v1"] = "feedback.frame.v1"

    frame_id: str
    generated_at: datetime

    source_execution_dispatch_frame_id: str
    source_policy_frame_id: str | None = None
    source_proposal_frame_id: str | None = None
    source_self_state_id: str | None = None

    feedback_policy_id: str = "feedback_policy.v1"

    outcome_status: Literal[
        "dry_run_only",
        "prepared_only",
        "completed",
        "failed",
        "blocked",
        "deferred",
        "absent",
        "mixed",
        "unknown",
    ] = "unknown"

    outcome_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    observations: list[OutcomeObservationV1] = Field(default_factory=list)

    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    absence_evidence: list[str] = Field(default_factory=list)

    pressure_before: dict[str, float] = Field(default_factory=dict)
    pressure_after: dict[str, float] = Field(default_factory=dict)
    pressure_delta: dict[str, float] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)
