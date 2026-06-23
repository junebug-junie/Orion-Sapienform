from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SelfStateConditionThresholdsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiet_max: float = 0.15
    steady_max: float = 0.40
    loaded_max: float = 0.70
    strained_max: float = 0.90


class SelfStatePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self_state_policy.v1"] = "self_state_policy.v1"
    policy_id: str = "self_state_policy.v1"

    condition_thresholds: SelfStateConditionThresholdsV1 = Field(
        default_factory=SelfStateConditionThresholdsV1
    )

    dimension_weights: dict[str, float] = Field(default_factory=dict)
    attention_target_weights: dict[str, float] = Field(default_factory=dict)
    channel_dimension_map: dict[str, str] = Field(default_factory=dict)
    stabilizing_channels: dict[str, float] = Field(default_factory=dict)
    pressure_channels: list[str] = Field(default_factory=list)
    context_channels: list[str] = Field(default_factory=list)

    unresolved_pressure_threshold: float = 0.60
    dominant_channel_threshold: float = 0.25
    trajectory_threshold: float = 0.03


def load_self_state_policy(path: str | Path) -> SelfStatePolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SelfStatePolicyV1.model_validate(data)
