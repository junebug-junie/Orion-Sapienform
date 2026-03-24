from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


WorkflowInvocationMode = Literal["immediate", "scheduled"]
WorkflowScheduleKind = Literal["one_shot", "recurring"]
WorkflowNotifyOn = Literal["none", "success", "failure", "completion"]
WorkflowCadence = Literal["daily", "weekly"]


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
