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
    # Evidence-only channel->dimension routing: channels here contribute to a
    # dimension's dominant_evidence/reasons transparency but NOT to its score
    # (that's channel_dimension_map's job). Exists for raw channels that are
    # also diffused into a capability channel under a different name -- the
    # diffused name stays in channel_dimension_map (so it alone feeds the
    # score, fixing the 2026-07-12 double-counting bug), while the raw name
    # stays visible here so downstream consumers can still see which real
    # signal underlies the dimension.
    evidence_channel_map: dict[str, str] = Field(default_factory=dict)
    stabilizing_channels: dict[str, float] = Field(default_factory=dict)
    pressure_channels: list[str] = Field(default_factory=list)
    context_channels: list[str] = Field(default_factory=list)
    # Per-dimension "worse" direction for the Phase 2 deviation probe
    # (orion/self_state/deviation.py): "up" if a rising score is the notable
    # direction, "down" if falling is. Config, not code, per this redesign's
    # own design invariant -- config/self_state/self_state_policy.v1.yaml is
    # the single place this fact lives, matching the house pattern already
    # established for the same concept in config/autonomy/signal_drive_map.yaml.
    dimension_worse_direction: dict[str, Literal["up", "down"]] = Field(default_factory=dict)

    unresolved_pressure_threshold: float = 0.60
    dominant_channel_threshold: float = 0.25
    trajectory_threshold: float = 0.03


def load_self_state_policy(path: str | Path) -> SelfStatePolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SelfStatePolicyV1.model_validate(data)
