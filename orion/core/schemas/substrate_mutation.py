"""Typed mutation contracts for Substrate adaptation V2.1."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

MutationLaneV1 = Literal["operational", "cognitive"]
MutationRiskTierV1 = Literal["low", "medium", "high"]
MutationLifecycleStateV1 = Literal["proposed", "queued", "trialed", "approved", "applied", "rolled_back", "rejected"]
MutationClassV1 = Literal[
    "routing_threshold_patch",
    "recall_weighting_patch",
    "graph_consolidation_param_patch",
    "approved_prompt_profile_variant_promotion",
]
MutationDecisionActionV1 = Literal["reject", "hold", "require_review", "auto_promote"]
MutationTrialStatusV1 = Literal["passed", "failed", "inconclusive"]


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
