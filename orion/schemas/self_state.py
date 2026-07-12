from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SelfStateDimensionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_id: Literal[
        "field_intensity",
        "coherence",
        "uncertainty",
        "agency_readiness",
        "resource_pressure",
        "execution_pressure",
        "reasoning_pressure",
        "reliability_pressure",
        "continuity_pressure",
        "introspection_pressure",
        "social_pressure",
        "transport_integrity",
    ]

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    dominant_evidence: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AttentionTargetSummaryV1(BaseModel):
    """Structured per-target attention data (2026-07-12, inner-state
    unification Phase 1). Previously orion/self_state/builder.py read the
    full FieldAttentionTargetV1 objects and kept only bare target_id strings
    on dominant_attention_targets -- pressure_score/dominant_channels/reasons
    were computed by orion-attention-runtime's real, non-theater scoring
    (weighted_pressure/urgency_score/confidence_from_vector) and discarded
    one hop downstream. This is additive alongside dominant_attention_targets,
    not a replacement -- existing consumers of the bare-string list are
    unaffected.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str
    target_kind: Literal["node", "capability", "channel", "edge", "field", "system"]
    pressure_score: float = Field(ge=0.0, le=1.0)
    dominant_channel: str | None = None
    reason: str | None = None


class SelfStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self.state.v1"] = "self.state.v1"

    self_state_id: str
    generated_at: datetime

    source_field_tick_id: str
    source_field_generated_at: datetime

    source_attention_frame_id: str
    source_attention_generated_at: datetime

    self_state_policy_id: str = "self_state_policy.v1"

    overall_condition: Literal[
        "quiet",
        "steady",
        "loaded",
        "strained",
        "unstable",
        "unknown",
    ] = "unknown"

    overall_intensity: float = Field(ge=0.0, le=1.0)
    overall_confidence: float = Field(ge=0.0, le=1.0)

    dimensions: dict[str, SelfStateDimensionV1] = Field(default_factory=dict)

    dominant_attention_targets: list[str] = Field(default_factory=list)
    dominant_attention_target_details: list[AttentionTargetSummaryV1] = Field(default_factory=list)
    dominant_field_channels: dict[str, float] = Field(default_factory=dict)

    unresolved_pressures: list[str] = Field(default_factory=list)
    stabilizing_factors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    summary_labels: list[str] = Field(default_factory=list)

    dimension_trajectory: dict[str, float] = Field(default_factory=dict)
    trajectory_condition: Literal["improving", "degrading", "stable", "unknown"] = "unknown"
    prediction_error_scores: dict[str, float] = Field(default_factory=dict)
    overall_surprise: float = Field(default=0.0, ge=0.0, le=1.0)

    # Attention schema: Orion's current focus quality and type
    attention_schema_type: Literal[
        "focused_single",
        "distributed",
        "open_loop",
        "none",
        "unknown",
    ] | None = None
    attention_dwell_ticks: int = 0
    attention_node_count: int = 0

    # Hub presence: Orion's connection liveness and chat rhythm
    hub_presence: dict[str, Any] | None = None
