from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProposalCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str

    proposal_kind: Literal[
        "observe",
        "inspect",
        "summarize",
        "stabilize",
        "defer",
        "request_policy_review",
        "prepare_action",
    ]

    title: str
    description: str

    target_id: str
    target_kind: Literal[
        "node",
        "capability",
        "field",
        "self_state",
        "service",
        "system",
    ]

    priority_score: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    reversibility_score: float = Field(ge=0.0, le=1.0)

    motivating_dimensions: dict[str, float] = Field(default_factory=dict)
    motivating_targets: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    proposed_effect: Literal[
        "increase_observability",
        "reduce_pressure",
        "preserve_stability",
        "increase_coherence",
        "defer_until_policy",
        "prepare_for_policy_gate",
        "no_effect",
    ]

    required_policy_gate: Literal[
        "none",
        "read_only",
        "operator_review",
        "autonomy_policy",
        "execution_policy",
    ] = "read_only"

    execution_intent: dict[str, str] = Field(default_factory=dict)

    # Provenance for candidates injected from outside the deterministic builder
    # (Phase B: spontaneous-thought proposals). None for builder-native candidates.
    source: str | None = None
    thought_id: str | None = None


class ProposalFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["proposal.frame.v1"] = "proposal.frame.v1"

    frame_id: str
    generated_at: datetime

    source_self_state_id: str
    source_self_state_generated_at: datetime

    source_attention_frame_id: str
    source_field_tick_id: str

    proposal_policy_id: str = "proposal_policy.v1"

    overall_action_pressure: float = Field(ge=0.0, le=1.0)
    overall_risk: float = Field(ge=0.0, le=1.0)
    policy_required: bool = True

    candidates: list[ProposalCandidateV1] = Field(default_factory=list)
    suppressed_candidates: list[ProposalCandidateV1] = Field(default_factory=list)

    dominant_motivations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
