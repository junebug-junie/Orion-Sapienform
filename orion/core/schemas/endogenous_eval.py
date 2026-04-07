"""Offline endogenous runtime evaluation and calibration contracts (Phase 11)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EndogenousEvaluationRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"endogenous-eval-{uuid4()}")
    limit: int = Field(default=500, ge=1, le=5000)
    invocation_surfaces: list[str] = Field(default_factory=list)
    workflow_types: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    subject_ref: Optional[str] = None
    mentor_invoked: Optional[bool] = None
    created_after: Optional[datetime] = None
    min_sample_size: int = Field(default=30, ge=1, le=2000)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_after", "generated_at")
    @classmethod
    def _ensure_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)


class EndogenousMetricSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_size: int = 0
    by_surface: dict[str, int] = Field(default_factory=dict)
    by_workflow: dict[str, int] = Field(default_factory=dict)
    by_outcome: dict[str, int] = Field(default_factory=dict)
    trigger_rate: float = 0.0
    noop_rate: float = 0.0
    suppress_rate: float = 0.0
    failure_rate: float = 0.0
    cooldown_hit_rate: float = 0.0
    coalesce_rate: float = 0.0
    debounce_rate: float = 0.0
    mentor_selected_rate: float = 0.0
    mentor_invoked_rate: float = 0.0
    mentor_disabled_suppression_rate: float = 0.0
    contradiction_review_rate: float = 0.0
    concept_refinement_rate: float = 0.0
    autonomy_review_rate: float = 0.0
    reflective_journal_rate: float = 0.0
    materialized_artifact_avg: float = 0.0
    repeated_subject_density: float = 0.0


class PromotionCalibrationSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_size: int = 0
    blocked_like_rate: float = 0.0
    materialization_success_rate: float = 0.0
    recommendation: str = "hold"
    rationale: str = "insufficient_data"


class ReasoningSummaryCalibrationSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_size: int = 0
    fallback_proxy_rate: float = 0.0
    recommendation: str = "hold"
    rationale: str = "insufficient_data"


class EndogenousCalibrationRecommendationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    recommendation_id: str = Field(default_factory=lambda: f"endogenous-rec-{uuid4()}")
    target: Literal[
        "trigger_threshold",
        "workflow_cooldown",
        "mentor_gating",
        "summary_inclusion_threshold",
        "promotion_threshold",
        "workflow_allowlist",
    ]
    parameter: str
    current_value: str
    recommended_value: str
    direction: Literal["increase", "decrease", "hold"]
    confidence: Literal["low", "medium", "high"] = "low"
    rationale: str
    advisory_only: bool = True


class EndogenousCalibrationProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    profile_name: str
    generated_from_request_id: str
    overrides: dict[str, str] = Field(default_factory=dict)
    advisory_only: bool = True


class EndogenousEvaluationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request: EndogenousEvaluationRequestV1
    metrics: EndogenousMetricSummaryV1
    promotion: PromotionCalibrationSummaryV1
    reasoning_summary: ReasoningSummaryCalibrationSummaryV1
    recommendations: list[EndogenousCalibrationRecommendationV1] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_profile: Optional[EndogenousCalibrationProfileV1] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("generated_at")
    @classmethod
    def _ensure_generated_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
