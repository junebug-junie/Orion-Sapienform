"""Typed runtime audit/persistence contracts for endogenous workflow adoption (Phase 9)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .endogenous import EndogenousTriggerDecisionV1, EndogenousTriggerRequestV1, EndogenousWorkflowPlanV1


class EndogenousRuntimeSignalDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_summary_request_id: Optional[str] = None
    contradiction_count: int = 0
    contradiction_refs: list[str] = Field(default_factory=list)
    unresolved_contradiction_artifact_ids: list[str] = Field(default_factory=list)
    lifecycle_state: Literal["active", "dormant", "retired", "unknown"] = "unknown"
    spark_pressure: float = 0.0
    autonomy_pressure: float = 0.0
    concept_fragmentation_score: float = 0.0
    low_confidence_artifact_count: int = 0
    mentor_gap_count: int = 0
    selected_artifact_ids: list[str] = Field(default_factory=list)
    signal_sources: dict[str, str] = Field(default_factory=dict)


class EndogenousRuntimeExecutionRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    runtime_record_id: str = Field(default_factory=lambda: f"endogenous-runtime-{uuid4()}")
    invocation_surface: str
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    subject_ref: Optional[str] = None
    trigger_request: EndogenousTriggerRequestV1
    signal_digest: EndogenousRuntimeSignalDigestV1
    decision: EndogenousTriggerDecisionV1
    plan: EndogenousWorkflowPlanV1
    mentor_invoked: bool = False
    mentor_request_id: Optional[str] = None
    materialized_artifact_ids: list[str] = Field(default_factory=list)
    execution_success: bool = True
    execution_error: Optional[str] = None
    audit_events: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class EndogenousRuntimeAuditV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enabled: bool
    invocation_surface: Optional[str] = None
    status: Literal[
        "disabled",
        "not_selected",
        "surface_disabled",
        "sampled_out",
        "ok",
        "failed",
        "persist_failed",
    ]
    allow_mentor_branch: bool
    allowed_workflow_types: list[str] = Field(default_factory=list)
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    duration_ms: int = 0
    runtime_record_id: Optional[str] = None
    decision_outcome: Optional[str] = None
    workflow_type: Optional[str] = None
    cooldown_applied: bool = False
    debounce_applied: bool = False
    mentor_invoked: bool = False
    materialized_artifact_ids: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class EndogenousRuntimeResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    audit: EndogenousRuntimeAuditV1
    record: Optional[EndogenousRuntimeExecutionRecordV1] = None


class EndogenousRuntimeQueryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    limit: int = Field(default=25, ge=1, le=200)
    invocation_surface: Optional[str] = None
    workflow_type: Optional[str] = None
    outcome: Optional[str] = None
    subject_ref: Optional[str] = None
    mentor_invoked: Optional[bool] = None
    created_after: Optional[datetime] = None

    @field_validator("created_after")
    @classmethod
    def _ensure_query_tz(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)


class EndogenousRuntimeConsumptionItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    runtime_record_id: str
    subject_ref: Optional[str] = None
    invocation_surface: str
    workflow_type: str
    outcome: str
    mentor_invoked: bool = False
    reasons: list[str] = Field(default_factory=list)
    materialized_artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime
