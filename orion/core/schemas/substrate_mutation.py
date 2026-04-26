"""Typed mutation contracts for Substrate adaptation V2.1."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

MutationLaneV1 = Literal["operational", "cognitive"]
MutationRiskTierV1 = Literal["low", "medium", "high"]
MutationLifecycleStateV1 = Literal[
    "proposed",
    "queued",
    "trialed",
    "pending_review",
    "accepted_as_draft",
    "approved",
    "applied",
    "rolled_back",
    "rejected",
    "superseded",
    "archived",
]
MutationClassV1 = Literal[
    "routing_threshold_patch",
    "recall_weighting_patch",
    "recall_strategy_profile_candidate",
    "recall_anchor_policy_candidate",
    "recall_page_index_profile_candidate",
    "recall_graph_expansion_policy_candidate",
    "graph_consolidation_param_patch",
    "approved_prompt_profile_variant_promotion",
    "cognitive_contradiction_reconciliation",
    "cognitive_identity_continuity_adjustment",
    "cognitive_stance_continuity_adjustment",
    "cognitive_social_continuity_repair",
]
MutationDecisionActionV1 = Literal["reject", "hold", "require_review", "auto_promote"]
MutationTrialStatusV1 = Literal["passed", "failed", "inconclusive"]
MutationPressureCategoryV1 = Literal[
    "routing_false_escalation",
    "routing_false_downgrade",
    "recall_miss_or_dissatisfaction",
    "unsupported_memory_claim",
    "irrelevant_semantic_neighbor",
    "missing_exact_anchor",
    "stale_memory_selected",
    "response_truncation_or_length_finish",
    "runtime_degradation_or_timeout",
    "social_addressedness_gap",
]
RecallStrategyProfileStatusV1 = Literal["draft", "staged", "shadow_active", "rejected", "archived"]
RecallShadowEvalRunStatusV1 = Literal["completed", "failed", "dry_run"]
RecallProductionCandidateRecommendationV1 = Literal[
    "keep_shadowing",
    "expand_shadow_corpus",
    "ready_for_manual_canary",
    "reject_candidate",
]
RecallProductionCandidateReviewStatusV1 = Literal["draft", "reviewed", "rejected", "archived"]
RecallCanaryJudgmentV1 = Literal["v2_better", "v1_better", "tie", "both_bad", "inconclusive"]
RecallCanaryFailureModeV1 = Literal[
    "missing_exact_anchor",
    "irrelevant_semantic_neighbor",
    "stale_memory",
    "unsupported_memory_claim",
    "insufficient_context",
    "wrong_project",
    "wrong_timeframe",
    "empty_result",
    "overbroad_result",
]


class MutationPressureEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pressure_event_id: str = Field(default_factory=lambda: f"substrate-pressure-event-{uuid4()}")
    source_service: str
    source_event_id: str
    correlation_id: str | None = None
    pressure_category: MutationPressureCategoryV1
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list, min_length=1, max_length=32)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class MutationSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(default_factory=lambda: f"substrate-mutation-signal-{uuid4()}")
    event_kind: str
    anchor_scope: str
    subject_ref: str
    target_surface: str
    target_zone: str = "concept_graph"
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list, min_length=1, max_length=32)
    source_ref: str = ""
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class MutationPressureV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pressure_id: str = Field(default_factory=lambda: f"substrate-mutation-pressure-{uuid4()}")
    anchor_scope: str
    subject_ref: str
    target_surface: str
    target_zone: str = "concept_graph"
    pressure_kind: str = "runtime_drift"
    pressure_score: float = Field(default=0.0, ge=0.0, le=100.0)
    evidence_refs: list[str] = Field(default_factory=list, min_length=1, max_length=64)
    source_signal_ids: list[str] = Field(default_factory=list, min_length=1, max_length=64)
    cooldown_until: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Last-merged structured recall shadow/compare payload for proposal-only recall candidates.
    recall_evidence_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Bounded FIFO of contributing recall shadow / eval-suite evidence (newest last).
    recall_evidence_history: list[dict[str, Any]] = Field(default_factory=list, max_length=12)


class MutationPatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mutation_class: MutationClassV1
    target_surface: str
    target_ref: str
    patch: dict[str, Any] = Field(default_factory=dict)
    rollback_payload: dict[str, Any] = Field(default_factory=dict)


class MutationProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(default_factory=lambda: f"substrate-mutation-proposal-{uuid4()}")
    lane: MutationLaneV1 = "operational"
    mutation_class: MutationClassV1
    risk_tier: MutationRiskTierV1
    target_surface: str
    anchor_scope: str
    subject_ref: str
    rationale: str = ""
    expected_effect: str = ""
    evidence_refs: list[str] = Field(default_factory=list, min_length=1, max_length=64)
    source_signal_ids: list[str] = Field(default_factory=list, min_length=1, max_length=64)
    source_pressure_id: str
    patch: MutationPatchV1
    rollout_state: MutationLifecycleStateV1 = "proposed"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _validate_surface_alignment(self) -> "MutationProposalV1":
        if self.patch.target_surface != self.target_surface:
            raise ValueError("patch_target_surface_mismatch")
        if self.patch.rollback_payload == {}:
            raise ValueError("rollback_payload_required")
        return self


class MutationQueueItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_item_id: str = Field(default_factory=lambda: f"substrate-mutation-queue-{uuid4()}")
    proposal_id: str
    mutation_class: MutationClassV1
    target_surface: str
    priority: int = Field(default=50, ge=0, le=100)
    status: MutationLifecycleStateV1 = "queued"
    due_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MutationTrialV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(default_factory=lambda: f"substrate-mutation-trial-{uuid4()}")
    proposal_id: str
    mutation_class: MutationClassV1
    replay_corpus_id: str
    baseline_metric_ref: str
    status: MutationTrialStatusV1
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list, max_length=64)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MutationDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: f"substrate-mutation-decision-{uuid4()}")
    proposal_id: str
    action: MutationDecisionActionV1
    reason: str = ""
    requires_operator_review: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = Field(default_factory=list, max_length=64)


class MutationAdoptionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adoption_id: str = Field(default_factory=lambda: f"substrate-mutation-adoption-{uuid4()}")
    proposal_id: str
    decision_id: str
    target_surface: str
    applied_patch: dict[str, Any] = Field(default_factory=dict)
    rollback_payload: dict[str, Any] = Field(default_factory=dict, min_length=1)
    status: Literal["applied", "rolled_back"] = "applied"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rollback_window_sec: int = Field(default=900, ge=30, le=86400)


class MutationRollbackV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollback_id: str = Field(default_factory=lambda: f"substrate-mutation-rollback-{uuid4()}")
    adoption_id: str
    proposal_id: str
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict, min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CognitiveProposalReviewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(default_factory=lambda: f"substrate-mutation-cognitive-review-{uuid4()}")
    proposal_id: str
    state: Literal["pending_review", "accepted_as_draft", "rejected", "superseded", "archived"]
    reviewer: str = "operator"
    rationale: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = Field(default_factory=list, max_length=64)


class CognitiveDraftRecommendationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str = Field(default_factory=lambda: f"substrate-mutation-cognitive-draft-{uuid4()}")
    proposal_id: str
    mutation_class: MutationClassV1
    affected_surface: str
    pressure_kind: str
    evidence_refs: list[str] = Field(default_factory=list, min_length=1, max_length=64)
    suggested_operator_action: str
    blast_radius: str
    risk_tier: MutationRiskTierV1
    reversible_recommendation: str = "draft_only_recommendation_not_applied"
    status: Literal["draft_only_not_applied"] = "draft_only_not_applied"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = Field(default_factory=list, max_length=64)


class CognitiveProposalDraftV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str = Field(default_factory=lambda: f"substrate-mutation-cognitive-proposal-draft-{uuid4()}")
    proposal_id: str
    proposal_class: MutationClassV1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Literal["operator_review"] = "operator_review"
    state: Literal["active_draft", "archived", "superseded"] = "active_draft"
    title: str = ""
    summary: str = ""
    draft_content: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list, max_length=128)
    review_refs: list[str] = Field(default_factory=list, max_length=128)
    safety_scope: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, max_length=64)


class CognitiveStanceNoteV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance_note_id: str = Field(default_factory=lambda: f"substrate-mutation-cognitive-stance-note-{uuid4()}")
    source_proposal_id: str
    source_draft_id: str | None = None
    proposal_class: MutationClassV1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    visibility: Literal["metacog_only", "stance_and_metacog"] = "metacog_only"
    status: Literal["active", "archived", "expired"] = "active"
    ttl_turns: int = Field(default=20, ge=1, le=200)
    summary: str = ""
    note: str = ""
    evidence_refs: list[str] = Field(default_factory=list, max_length=128)
    review_ref: str | None = None
    safety_scope: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, max_length=64)


class RecallStrategyProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(default_factory=lambda: f"substrate-recall-strategy-profile-{uuid4()}")
    source_proposal_id: str
    source_pressure_ids: list[str] = Field(default_factory=list, max_length=64)
    source_evidence_refs: list[str] = Field(default_factory=list, max_length=128)
    readiness_snapshot: dict[str, Any] = Field(default_factory=dict)
    strategy_kind: str
    recall_v2_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    anchor_policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    page_index_policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    graph_expansion_policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    eval_evidence_refs: list[str] = Field(default_factory=list, max_length=128)
    created_by: str = "operator"
    status: RecallStrategyProfileStatusV1 = "draft"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecallShadowEvalRunV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"substrate-recall-shadow-eval-run-{uuid4()}")
    profile_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dry_run: bool = True
    recorded_pressure_events: int = Field(default=0, ge=0, le=2048)
    corpus_limit: int = Field(default=24, ge=1, le=512)
    case_ids: list[str] = Field(default_factory=list, max_length=256)
    eval_row_count: int = Field(default=0, ge=0, le=10000)
    readiness_before: dict[str, Any] = Field(default_factory=dict)
    readiness_after: dict[str, Any] = Field(default_factory=dict)
    readiness_delta_summary: dict[str, Any] = Field(default_factory=dict)
    pressure_event_refs: list[str] = Field(default_factory=list, max_length=512)
    operator_rationale: str = ""
    status: RecallShadowEvalRunStatusV1 = "dry_run"
    failure_reason: str | None = None


class RecallProductionCandidateReviewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(default_factory=lambda: f"substrate-recall-production-candidate-review-{uuid4()}")
    profile_id: str
    source_eval_run_ids: list[str] = Field(default_factory=list, max_length=256)
    readiness_snapshot: dict[str, Any] = Field(default_factory=dict)
    risk_summary: list[str] = Field(default_factory=list, max_length=64)
    observed_improvements: list[str] = Field(default_factory=list, max_length=64)
    observed_regressions: list[str] = Field(default_factory=list, max_length=64)
    operator_checklist: dict[str, Any] = Field(default_factory=dict)
    recommendation: RecallProductionCandidateRecommendationV1 = "keep_shadowing"
    status: RecallProductionCandidateReviewStatusV1 = "draft"
    created_by: str = "operator"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecallCanaryRunV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canary_run_id: str = Field(default_factory=lambda: f"substrate-recall-canary-run-{uuid4()}")
    profile_id: str | None = None
    query_text: str
    query_profile: str | None = None
    source: str = "operator_manual_canary"
    comparison_summary: dict[str, Any] = Field(default_factory=dict)
    v1_summary: dict[str, Any] = Field(default_factory=dict)
    v2_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecallCanaryJudgmentRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgment_id: str = Field(default_factory=lambda: f"substrate-recall-canary-judgment-{uuid4()}")
    canary_run_id: str
    profile_id: str | None = None
    query_text: str = ""
    judgment: RecallCanaryJudgmentV1
    failure_modes: list[RecallCanaryFailureModeV1] = Field(default_factory=list, max_length=32)
    operator_note: str = ""
    should_emit_pressure: bool = True
    should_mark_review_candidate: bool = False
    pressure_event_refs: list[str] = Field(default_factory=list, max_length=64)
    review_candidate_marked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecallCanaryReviewArtifactV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_artifact_id: str = Field(default_factory=lambda: f"substrate-recall-canary-review-artifact-{uuid4()}")
    canary_run_id: str
    profile_id: str | None = None
    linked_review_id: str | None = None
    review_type: str = "production_candidate_evidence"
    include_comparison_summary: bool = True
    include_operator_judgment: bool = True
    operator_note: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
