"""Telemetry and advisory calibration contracts for bounded review runtime (Phase 12)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.core.schemas.substrate_mutation import MutationPressureEvidenceV1
from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeOutcomeV1, GraphReviewRuntimeSurfaceV1

GraphReviewCalibrationRecommendationTypeV1 = Literal[
    "increase_cadence_interval",
    "decrease_cadence_interval",
    "increase_max_cycles",
    "decrease_max_cycles",
    "increase_suppression_threshold",
    "decrease_suppression_threshold",
    "hold",
]


class GraphReviewTelemetryRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telemetry_id: str = Field(default_factory=lambda: f"graph-review-telemetry-{uuid4()}")
    correlation_id: Optional[str] = None
    policy_profile_id: Optional[str] = None
    invocation_surface: GraphReviewRuntimeSurfaceV1
    queue_item_id: Optional[str] = None
    anchor_scope: Optional[str] = None
    subject_ref: Optional[str] = None
    target_zone: Optional[FrontierTargetZoneV1] = None
    selected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    selection_reason: str
    selected_priority: Optional[int] = Field(default=None, ge=0, le=100)
    cycle_count_before: Optional[int] = Field(default=None, ge=0)
    cycle_count_after: Optional[int] = Field(default=None, ge=0)
    remaining_cycles_before: Optional[int] = Field(default=None, ge=0)
    remaining_cycles_after: Optional[int] = Field(default=None, ge=0)
    consolidation_outcomes: List[str] = Field(default_factory=list, max_length=16)
    suppression_state_before: Optional[bool] = None
    suppression_state_after: Optional[bool] = None
    termination_state_before: Optional[bool] = None
    termination_state_after: Optional[bool] = None
    frontier_followup_invoked: bool = False
    execution_outcome: GraphReviewRuntimeOutcomeV1
    runtime_duration_ms: int = Field(ge=0)
    notes: List[str] = Field(default_factory=list, max_length=64)
    degraded: bool = False
    pressure_events: List[MutationPressureEvidenceV1] = Field(default_factory=list, max_length=16)


class GraphReviewTelemetryQueryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_surface: Optional[GraphReviewRuntimeSurfaceV1] = None
    target_zone: Optional[FrontierTargetZoneV1] = None
    subject_ref: Optional[str] = None
    outcome: Optional[GraphReviewRuntimeOutcomeV1] = None
    frontier_followup_invoked: Optional[bool] = None
    since: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=500)


class GraphReviewTelemetrySummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_executions: int = Field(ge=0)
    total_noops: int = Field(ge=0)
    total_suppressed: int = Field(ge=0)
    total_terminated: int = Field(ge=0)
    total_failed: int = Field(ge=0)
    outcome_counts: Dict[str, int] = Field(default_factory=dict)
    zone_counts: Dict[str, int] = Field(default_factory=dict)
    surface_counts: Dict[str, int] = Field(default_factory=dict)
    frontier_followup_counts: Dict[str, int] = Field(default_factory=dict)
    avg_cycles_before_resolution: float = Field(ge=0.0)
    avg_runtime_duration_ms: float = Field(ge=0.0)
    query_metadata: Dict[str, object] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list, max_length=32)


class GraphReviewCalibrationRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_sample_size: int = Field(default=8, ge=1, le=500)
    high_suppression_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    high_requeue_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    high_failure_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class GraphReviewCalibrationRecommendationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(default_factory=lambda: f"graph-review-calibration-{uuid4()}")
    recommendation_type: GraphReviewCalibrationRecommendationTypeV1
    target_parameter: str
    current_value: str
    suggested_value: str
    rationale: str
    sample_size: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    affected_zone: Optional[FrontierTargetZoneV1] = None
    affected_surface: Optional[GraphReviewRuntimeSurfaceV1] = None
    notes: List[str] = Field(default_factory=list, max_length=32)
