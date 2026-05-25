from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FeedbackWindowsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_after_window_sec: int = 30
    result_wait_window_sec: int = 30
    stale_after_sec: int = 120


class FeedbackScoringV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dry_run_score: float = 0.50
    prepared_score: float = 0.55
    completed_score: float = 0.85
    blocked_score: float = 0.40
    deferred_score: float = 0.45
    failed_score: float = 0.10
    absent_score: float = 0.15
    unknown_score: float = 0.25


class FeedbackPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["feedback_policy.v1"] = "feedback_policy.v1"
    policy_id: str = "feedback_policy.v1"

    windows: FeedbackWindowsV1 = Field(default_factory=FeedbackWindowsV1)
    scoring: FeedbackScoringV1 = Field(default_factory=FeedbackScoringV1)
    pressure_channels: list[str] = Field(default_factory=list)
    positive_delta_channels: dict[str, str] = Field(default_factory=dict)
    absence_rules: dict[str, bool | str] = Field(default_factory=dict)


def load_feedback_policy(path: str | Path) -> FeedbackPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return FeedbackPolicyV1.model_validate(data)
