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
        # 2026-08-12: a bounded, reversible mutating action, approved. This
        # exists because the alternative was a lie. Before it, a `maintain`
        # candidate was approved as `approved_read_only` -- the only value
        # `allowed_policy_decisions` accepted -- with the persisted reason
        # `read_only_low_risk`, so the audit record for deleting ~142 GB of
        # host files stated that the action reads nothing. Per CLAUDE.md
        # §0A ("if Orion says it decided, there must be inspectable evidence
        # for that claim"), a decision record whose stated grounds are false
        # is the defect; the enum name was the smaller half of it.
        #
        # It also fixed a live false positive: orion/consolidation/motif.py
        # counts `approved_read_only` decisions to detect a
        # "read_only_policy_loop" motif, and every maintain approval was
        # feeding that detector a sample it should never have seen.
        #
        # This literal DESCRIBES, it does not authorize. The real gate stays
        # orion/execution_dispatch/builder.py's scope check.
        "approved_maintenance",
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
        # 2026-08-12: the tier a `maintain` candidate is evaluated at.
        # Deliberately NOT "execution_allowed": that tier is the open-ended
        # one this substrate has never granted. `maintenance_bounded` names a
        # narrower thing -- a reversible housekeeping action on a route whose
        # scope is checked by name in orion/execution_dispatch/builder.py.
        "maintenance_bounded",
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
        # 2026-08-12: MISSING, and that made the whole maintenance chain dead
        # on arrival -- worse than dead. config/policy/substrate_policy.v1.yaml's
        # `maintain` rule sets allowed_scope AND max_autonomy_tier to
        # "maintenance_bounded", so evaluate_proposal_candidate() built a
        # PolicyDecisionV1 that failed validation on both fields. That raises
        # during decision *construction*, which
        # services/orion-policy-runtime/app/store.py's
        # _retire_incompatible_proposal_frame does NOT cover (it only catches a
        # proposal frame that fails to LOAD). The cursor would never advance
        # past the first prune proposal: not a no-op, a permanent stall of the
        # entire policy runtime. Caught in review before merge; reproduced
        # live, then fixed. See test_maintain_candidate_survives_policy_evaluation.
        "maintenance_bounded",
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
    # source_self_state_id removed 2026-07-22, SelfStateV1 burn -- redundant
    # with source_field_tick_id, which already independently identified the
    # same tick.
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
