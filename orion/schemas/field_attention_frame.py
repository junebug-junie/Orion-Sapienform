from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldAttentionTargetV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    target_kind: Literal[
        "node",
        "capability",
        "channel",
        "edge",
        "field",
        "system",
    ]

    salience_score: float = Field(ge=0.0, le=1.0)
    pressure_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    dominant_channels: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    suggested_observation_mode: Literal[
        "watch",
        "inspect",
        "summarize",
        "ignore",
    ] = "watch"


class FieldAttentionFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["field.attention.frame.v1"] = "field.attention.frame.v1"

    frame_id: str
    generated_at: datetime

    source_field_tick_id: str
    source_field_generated_at: datetime

    attention_policy_id: str = "field_attention_policy.v1"

    overall_salience: float = Field(ge=0.0, le=1.0)

    dominant_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    node_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    capability_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    system_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    suppressed_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)

    recent_perturbations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
