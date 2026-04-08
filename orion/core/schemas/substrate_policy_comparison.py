"""Policy comparison and rollout effectiveness contracts (post-Phase 19)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SubstratePolicyMetricDeltaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline_value: float
    candidate_value: float
    delta: float
    pct_delta: float


class SubstratePolicyComparisonRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"substrate-policy-compare-{uuid4()}")
    candidate_profile_id: Optional[str] = None
    baseline_profile_id: Optional[str] = None
    baseline_window_label: str = "baseline_window"
    candidate_window_label: str = "candidate_window"
    operator_id: Optional[str] = None
    rationale: str = ""


class SubstratePolicyEffectivenessReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    candidate_profile_id: Optional[str] = None
    baseline_profile_id: Optional[str] = None
    baseline_window_label: str
    candidate_window_label: str
    verdict: Literal["improved", "neutral", "degraded", "insufficient_data"]
    confidence: float = Field(ge=0.0, le=1.0)
    metric_deltas: list[SubstratePolicyMetricDeltaV1] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, max_length=64)
    compared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
