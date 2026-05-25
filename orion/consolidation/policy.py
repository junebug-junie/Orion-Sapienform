from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConsolidationWindowConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback_minutes: int = 60
    min_support_count: int = 3
    max_frames_per_source: int = 500


class MotifThresholdsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_support_score: float = 0.20
    min_confidence_score: float = 0.30
    dominant_motif_min_support: float = 0.50


class MotifRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "field_pattern",
        "attention_pattern",
        "self_state_pattern",
        "proposal_policy_pattern",
        "dispatch_feedback_pattern",
        "absence_pattern",
        "stability_pattern",
    ]
    label: str
    conditions: dict[str, Any] = Field(default_factory=dict)


class TensorConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_coordinates: int = 200


class ConsolidationPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["consolidation_policy.v1"] = "consolidation_policy.v1"
    policy_id: str = "consolidation_policy.v1"

    window: ConsolidationWindowConfigV1 = Field(default_factory=ConsolidationWindowConfigV1)
    motif_thresholds: MotifThresholdsV1 = Field(default_factory=MotifThresholdsV1)

    tracked_self_dimensions: list[str] = Field(default_factory=list)
    tracked_attention_targets: list[str] = Field(default_factory=list)
    tracked_feedback_outcomes: list[str] = Field(default_factory=list)

    motif_rules: dict[str, MotifRuleV1] = Field(default_factory=dict)

    tensor: TensorConfigV1 = Field(default_factory=TensorConfigV1)
    tensor_axes: dict[str, list[str]] = Field(default_factory=dict)


def load_consolidation_policy(path: str | Path) -> ConsolidationPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw_rules = data.pop("motif_rules", {}) or {}
    rules = {key: MotifRuleV1.model_validate(val) for key, val in raw_rules.items()}
    base = ConsolidationPolicyV1.model_validate(data)
    return base.model_copy(update={"motif_rules": rules})
