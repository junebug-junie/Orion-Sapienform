from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    proposal_id: str

    decision: Literal[
        "approved_for_execution",
        "approved_read_only",
        "requires_operator_review",
        "deferred",
        "rejected",
    ]

    policy_gate: Literal[
        "none",
        "read_only",
        "operator_review",
        "autonomy_policy",
        "execution_policy",
    ]

    autonomy_tier: Literal[
        "none",
        "observe_only",
        "read_only",
        "prepare_only",
        "operator_review",
        "execution_allowed",
    ] = "observe_only"

    risk_score: float = Field(ge=0.0, le=1.0)
    reversibility_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    allowed_scope: Literal[
        "none",
        "inspect_only",
        "summarize_only",
        "prepare_only",
        "low_risk_execution",
        "operator_review_required",
    ] = "none"

    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)

    execution_constraints: dict[str, str] = Field(default_factory=dict)


class PolicyDecisionFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["policy.decision.frame.v1"] = "policy.decision.frame.v1"

    frame_id: str
    generated_at: datetime

    source_proposal_frame_id: str
    source_self_state_id: str
    source_attention_frame_id: str | None = None
    source_field_tick_id: str | None = None

    policy_id: str = "substrate_policy.v1"

    decisions: list[PolicyDecisionV1] = Field(default_factory=list)
    approved_decisions: list[PolicyDecisionV1] = Field(default_factory=list)
    review_required_decisions: list[PolicyDecisionV1] = Field(default_factory=list)
    deferred_decisions: list[PolicyDecisionV1] = Field(default_factory=list)
    rejected_decisions: list[PolicyDecisionV1] = Field(default_factory=list)

    overall_risk: float = Field(ge=0.0, le=1.0)
    operator_review_required: bool = False
    execution_allowed: bool = False

    warnings: list[str] = Field(default_factory=list)
