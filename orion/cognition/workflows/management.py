from __future__ import annotations

import re
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .registry import resolve_user_workflow_invocation
from orion.schemas.workflow_execution import WorkflowScheduleManageRequestV1, WorkflowScheduleUpdatePatchV1


class WorkflowScheduleManagementIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: WorkflowScheduleManageRequestV1


def _normalize(prompt: str) -> str:
    return re.sub(r"\s+", " ", (prompt or "").strip().lower())


def _parse_time(prompt: str) -> tuple[int, int] | None:
    m = re.search(r"(?:to|at)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", prompt)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    meridiem = (m.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def resolve_workflow_schedule_management(prompt: str, *, user_id: str | None, session_id: str | None) -> Optional[WorkflowScheduleManagementIntent]:
    normalized = _normalize(prompt)
    if not normalized:
        return None

    workflow = resolve_user_workflow_invocation(prompt)
    workflow_id = workflow.workflow_id if workflow else None
    req_id = str(uuid4())

    if any(phrase in normalized for phrase in ("what workflow runs", "what do i have scheduled", "list scheduled", "show scheduled")):
        return WorkflowScheduleManagementIntent(
            request=WorkflowScheduleManageRequestV1(
                operation="list",
                request_id=req_id,
                workflow_id=workflow_id,
                include_history=("history" in normalized or "status" in normalized),
                origin_user_id=user_id,
                session_id=session_id,
            )
        )

    if normalized.startswith("cancel") and ("schedule" in normalized or "every" in normalized or workflow_id):
        return WorkflowScheduleManagementIntent(
            request=WorkflowScheduleManageRequestV1(
                operation="cancel",
                request_id=req_id,
                workflow_id=workflow_id,
                origin_user_id=user_id,
                session_id=session_id,
            )
        )

    if normalized.startswith("pause") and ("schedule" in normalized or workflow_id):
        return WorkflowScheduleManagementIntent(
            request=WorkflowScheduleManageRequestV1(
                operation="pause",
                request_id=req_id,
                workflow_id=workflow_id,
                origin_user_id=user_id,
                session_id=session_id,
            )
        )

    if normalized.startswith("resume") and ("schedule" in normalized or workflow_id):
        return WorkflowScheduleManagementIntent(
            request=WorkflowScheduleManageRequestV1(
                operation="resume",
                request_id=req_id,
                workflow_id=workflow_id,
                origin_user_id=user_id,
                session_id=session_id,
            )
        )

    if any(token in normalized for token in ("move ", "reschedule", "change ")) and workflow_id:
        parsed_time = _parse_time(normalized)
        patch = WorkflowScheduleUpdatePatchV1()
        if parsed_time:
            patch.hour_local, patch.minute_local = parsed_time
        if "every friday" in normalized:
            patch.cadence = "weekly"
            patch.day_of_week = 4
        elif "nightly" in normalized or "every night" in normalized:
            patch.cadence = "daily"
        return WorkflowScheduleManagementIntent(
            request=WorkflowScheduleManageRequestV1(
                operation="update",
                request_id=req_id,
                workflow_id=workflow_id,
                patch=patch,
                origin_user_id=user_id,
                session_id=session_id,
            )
        )

    return None
