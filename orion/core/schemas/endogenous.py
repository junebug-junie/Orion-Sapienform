"""Endogenous trigger orchestration contracts (Phase 7)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .reasoning import AnchorScope
from .reasoning_summary import ReasoningSummaryV1

EndogenousTriggerOutcomeV1 = Literal["trigger", "suppress", "defer", "coalesce", "noop"]
EndogenousWorkflowTypeV1 = Literal[
    "contradiction_review",
    "concept_refinement",
    "autonomy_review",
    "mentor_critique",
    "reflective_journal",
    "no_action",
]
EndogenousActionTypeV1 = Literal[
    "compile_context_slice",
    "run_contradiction_check",
    "run_concept_refinement",
    "review_autonomy_state",
    "invoke_mentor_gateway",
    "materialize_advisory_proposals",
    "promotion_gate_check",
    "emit_reflective_journal",
    "emit_audit_trace",
    "stop",
]


class EndogenousHistoryEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workflow_type: EndogenousWorkflowTypeV1
    subject_ref: Optional[str] = None
    cause_signature: str
    outcome: EndogenousTriggerOutcomeV1
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("recorded_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class EndogenousTriggerRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"endogenous-trigger-{uuid4()}")
    anchor_scope: AnchorScope = "orion"
    subject_ref: Optional[str] = None
    reasoning_summary: Optional[ReasoningSummaryV1] = None
    selected_artifact_ids: list[str] = Field(default_factory=list)
    contradiction_refs: list[str] = Field(default_factory=list)
    unresolved_contradiction_count: int = Field(default=0, ge=0)
    contradiction_severity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    spark_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    spark_instability: float = Field(default=0.0, ge=0.0, le=1.0)
    autonomy_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_fragmentation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    low_confidence_artifact_count: int = Field(default=0, ge=0)
    mentor_gap_count: int = Field(default=0, ge=0)
    lifecycle_state: Literal["active", "dormant", "retired", "unknown"] = "unknown"
    recent_history: list[EndogenousHistoryEntryV1] = Field(default_factory=list)
    trigger_version: str = "phase7.v1"
    policy_version: str = "phase7.policy.v1"
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evaluated_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class EndogenousTriggerSignalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contradiction_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    autonomy_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    mentor_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    reflective_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    total_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    triggerable: bool = False
    counters: dict[str, int] = Field(default_factory=dict)


class EndogenousTriggerDebugV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    considered_workflows: list[str] = Field(default_factory=list)
    selected_workflow_score: float = 0.0
    suppression_reasons: list[str] = Field(default_factory=list)
    cooldown_seconds_remaining: int = 0
    cause_signature: str = ""
    policy_counters: dict[str, float] = Field(default_factory=dict)


class EndogenousTriggerDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    outcome: EndogenousTriggerOutcomeV1
    workflow_type: EndogenousWorkflowTypeV1 = "no_action"
    reasons: list[str] = Field(default_factory=list)
    alternatives_not_chosen: list[str] = Field(default_factory=list)
    cooldown_applied: bool = False
    debounce_applied: bool = False
    coalesced: bool = False
    signal: EndogenousTriggerSignalV1 = Field(default_factory=EndogenousTriggerSignalV1)
    debug: EndogenousTriggerDebugV1 = Field(default_factory=EndogenousTriggerDebugV1)


class EndogenousWorkflowActionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action_id: str = Field(default_factory=lambda: f"endo-action-{uuid4()}")
    action_type: EndogenousActionTypeV1
    params: dict = Field(default_factory=dict)


class EndogenousWorkflowPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    plan_id: str = Field(default_factory=lambda: f"endo-plan-{uuid4()}")
    request_id: str
    workflow_type: EndogenousWorkflowTypeV1 = "no_action"
    trigger_outcome: EndogenousTriggerOutcomeV1
    reasons: list[str] = Field(default_factory=list)
    actions: list[EndogenousWorkflowActionV1] = Field(default_factory=list)
    max_actions: int = Field(default=6, ge=0, le=20)
    coalesced_with: Optional[str] = None
    audit: dict = Field(default_factory=dict)


class EndogenousWorkflowExecutionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    decision: EndogenousTriggerDecisionV1
    plan: EndogenousWorkflowPlanV1
    executed: bool = False
    mentor_invoked: bool = False
    materialized_artifact_ids: list[str] = Field(default_factory=list)
    audit_events: list[str] = Field(default_factory=list)
