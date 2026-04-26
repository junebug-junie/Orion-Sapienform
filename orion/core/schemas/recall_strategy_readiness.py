"""Recall V2 / recall-strategy promotion readiness (advisory only; no live apply)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RecallStrategyReadinessRecommendationV1 = Literal[
    "not_ready",
    "review_candidate",
    "ready_for_shadow_expansion",
    "ready_for_operator_promotion",
]


class RecallStrategyReadinessV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_coverage: float = Field(ge=0.0, le=1.0, description="Share of reference eval corpus represented by eval-suite evidence.")
    precision_proxy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean V2 precision proxy when present; else mean delta-implied score bounded to [0,1].",
    )
    answer_support_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    irrelevant_cousin_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_time_match_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_delta_ms_mean: float = Field(
        default=0.0,
        description="Mean(V2_latency_ms - V1_latency_ms); positive means V2 slower.",
    )
    explainability_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_categories_improved: dict[str, int] = Field(default_factory=dict)
    failure_categories_regressed: dict[str, int] = Field(default_factory=dict)
    minimum_evidence_cases_required: int = Field(default=3, ge=1, le=64)
    evidence_observation_count: int = Field(default=0, ge=0, le=512)
    minimum_evidence_met: bool = Field(default=False)
    recommendation: RecallStrategyReadinessRecommendationV1
    readiness_notes: list[str] = Field(default_factory=list, max_length=24)
    gates_blocked: list[str] = Field(default_factory=list, max_length=16)
