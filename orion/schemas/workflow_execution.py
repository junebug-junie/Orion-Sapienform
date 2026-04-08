from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


WorkflowInvocationMode = Literal["immediate", "scheduled"]
WorkflowScheduleKind = Literal["one_shot", "recurring"]
WorkflowNotifyOn = Literal["none", "success", "failure", "completion"]
WorkflowCadence = Literal["daily", "weekly"]
WorkflowScheduleState = Literal["scheduled", "due", "dispatched", "completed", "failed", "cancelled", "paused"]
WorkflowScheduleResult = Literal["unknown", "completed", "failed", "cancelled", "dispatched"]
WorkflowScheduleHealth = Literal["healthy", "degraded", "failing", "paused", "idle", "cancelled"]
WorkflowScheduleManageErrorCode = Literal[
    "invalid_management_payload",
    "ambiguous_selection",
    "schedule_not_found",
    "already_cancelled",
    "already_paused",
    "unsupported_transition",
    "missing_patch",
    "invalid_patch",
    "schedule_policy_missing",
    "schedule_revision_conflict",
]


class WorkflowScheduleSpecV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: WorkflowScheduleKind
    timezone: str = "America/Denver"
    run_at_utc: Optional[datetime] = None
    cadence: Optional[WorkflowCadence] = None
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    hour_local: Optional[int] = Field(default=None, ge=0, le=23)
    minute_local: Optional[int] = Field(default=None, ge=0, le=59)
    label: Optional[str] = None


class WorkflowExecutionPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    invocation_mode: WorkflowInvocationMode = "immediate"
    schedule: Optional[WorkflowScheduleSpecV1] = None
    notify_on: WorkflowNotifyOn = "none"
    recipient_group: str = "juniper_primary"
    session_id: Optional[str] = None
    origin_user_id: Optional[str] = None
    policy_summary: Optional[str] = None
    requested_from_chat: bool = True


class WorkflowDispatchRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    workflow_id: str
    workflow_request: Dict[str, Any] = Field(default_factory=dict)
    execution_policy: WorkflowExecutionPolicyV1
    correlation_id: Optional[str] = None
    source_service: str = "orion-cortex-orch"
    source_kind: Literal["chat_schedule_request", "actions_scheduler"] = "chat_schedule_request"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowScheduleRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schedule_id: str
    request_id: str
    workflow_id: str
    workflow_display_name: Optional[str] = None
    workflow_request: Dict[str, Any] = Field(default_factory=dict)
    execution_policy: WorkflowExecutionPolicyV1
    notify_on: WorkflowNotifyOn = "none"
    source_service: str = "orion-cortex-orch"
    source_kind: Literal["chat_schedule_request", "actions_scheduler"] = "chat_schedule_request"
    source_correlation_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    last_result_status: WorkflowScheduleResult = "unknown"
    state: WorkflowScheduleState = "scheduled"
    revision: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    analytics: Optional["WorkflowScheduleAnalyticsV1"] = None


class WorkflowScheduleAnalyticsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    health: WorkflowScheduleHealth = "idle"
    needs_attention: bool = False
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    recent_run_count: int = 0
    recent_success_count: int = 0
    recent_failure_count: int = 0
    recent_outcomes: List[str] = Field(default_factory=list)
    most_recent_result_status: Optional[str] = None
    is_overdue: bool = False
    overdue_seconds: Optional[int] = None
    missed_run_count: int = 0
    history_window_runs: int = 5


class WorkflowScheduleRunRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    schedule_id: str
    workflow_id: str
    request_id: str
    status: WorkflowScheduleResult = "dispatched"
    dispatch_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowScheduleEventRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    kind: str
    schedule_id: str
    occurred_at: datetime
    extra: Dict[str, Any] = Field(default_factory=dict)


WorkflowScheduleManageOperation = Literal["list", "cancel", "update", "pause", "resume", "history"]


class WorkflowScheduleUpdatePatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_at_utc: Optional[datetime] = None
    cadence: Optional[WorkflowCadence] = None
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    hour_local: Optional[int] = Field(default=None, ge=0, le=23)
    minute_local: Optional[int] = Field(default=None, ge=0, le=59)
    timezone: Optional[str] = None
    notify_on: Optional[WorkflowNotifyOn] = None
    expected_revision: Optional[int] = Field(default=None, ge=0)


class WorkflowScheduleManageRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: WorkflowScheduleManageOperation
    request_id: str
    schedule_id: Optional[str] = None
    workflow_id: Optional[str] = None
    include_history: bool = False
    patch: Optional[WorkflowScheduleUpdatePatchV1] = None
    source_service: str = "orion-cortex-orch"
    source_correlation_id: Optional[str] = None
    origin_user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowScheduleManageResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    operation: WorkflowScheduleManageOperation
    request_id: str
    message: str
    schedule: Optional[WorkflowScheduleRecordV1] = None
    schedules: List[WorkflowScheduleRecordV1] = Field(default_factory=list)
    history: List[WorkflowScheduleRunRecordV1] = Field(default_factory=list)
    events: List[WorkflowScheduleEventRecordV1] = Field(default_factory=list)
    ambiguous: bool = False
    error_code: Optional[WorkflowScheduleManageErrorCode] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
