"""Deterministic promotion and lifecycle policy contracts (Phase 3)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .reasoning import PromotionDecisionV1, ReasoningStatus, RiskTier

PromotionOutcome = Literal[
    "promoted",
    "blocked",
    "rejected",
    "deprecated",
    "escalated_hitl",
    "no_change",
]
LifecycleAction = Literal["none", "emerge", "strengthen", "dormant", "decay", "retire", "revive"]
LifecycleState = Literal["emerging", "active", "dormant", "decaying", "retired"]


class ContradictionFindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contradiction_id: str
    severity: Literal["low", "medium", "high", "critical"]
    resolution_status: Literal["open", "under_review", "resolved", "deferred"]


class EntityLifecycleEvaluationRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    anchor_scope: Literal["orion", "juniper", "relationship", "world", "session"]
    subject_ref: Optional[str] = None
    current_state: Optional[LifecycleState] = None
    now: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("now")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class EntityLifecycleEvaluationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    anchor_scope: Literal["orion", "juniper", "relationship", "world", "session"]
    subject_ref: Optional[str] = None
    prior_state: Optional[LifecycleState] = None
    next_state: Optional[LifecycleState] = None
    lifecycle_action: LifecycleAction = "none"
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PromotionEvaluationRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"promotion-eval-{uuid4()}")
    artifact_ids: list[str] = Field(min_length=1)
    target_status: ReasoningStatus
    actor: str = "reasoning-policy"
    policy_version: str = "phase3.v1"


class PromotionEvaluationItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_id: str
    artifact_type: str
    current_status: ReasoningStatus
    target_status: ReasoningStatus
    outcome: PromotionOutcome
    reasons: list[str] = Field(default_factory=list)
    risk_tier: RiskTier
    contradiction_findings: list[ContradictionFindingV1] = Field(default_factory=list)
    human_review_required: bool = False
    escalation_reason: Optional[str] = None
    lifecycle: Optional[EntityLifecycleEvaluationResultV1] = None
    policy_version: str = "phase3.v1"


class PromotionEvaluationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    policy_version: str
    evaluated_count: int
    items: list[PromotionEvaluationItemV1] = Field(default_factory=list)
    decisions: list[PromotionDecisionV1] = Field(default_factory=list)
    accepted_count: int = 0
    blocked_count: int = 0
    escalated_count: int = 0
    rejected_count: int = 0
