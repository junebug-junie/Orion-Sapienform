from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AttentionLimitsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_targets_total: int = 12
    max_node_targets: int = 5
    max_capability_targets: int = 5
    max_system_targets: int = 3


class AttentionThresholdsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_salience: float = 0.10
    high_salience: float = 0.70
    suppress_below: float = 0.03


class AttentionWeightsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pressure: float = 0.45
    novelty: float = 0.20
    urgency: float = 0.25
    confidence: float = 0.10


class ObservationModesV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inspect_threshold: float = 0.75
    summarize_threshold: float = 0.45
    watch_threshold: float = 0.10


class FieldAttentionPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["attention_policy.v1"] = "attention_policy.v1"
    policy_id: str = "field_attention_policy.v1"

    limits: AttentionLimitsV1 = Field(default_factory=AttentionLimitsV1)
    thresholds: AttentionThresholdsV1 = Field(default_factory=AttentionThresholdsV1)
    weights: AttentionWeightsV1 = Field(default_factory=AttentionWeightsV1)
    node_channel_weights: dict[str, float] = Field(default_factory=dict)
    capability_channel_weights: dict[str, float] = Field(default_factory=dict)
    observation_modes: ObservationModesV1 = Field(default_factory=ObservationModesV1)


def load_attention_policy(path: str | Path) -> FieldAttentionPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return FieldAttentionPolicyV1.model_validate(data)
