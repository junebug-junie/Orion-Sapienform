from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
from zoneinfo import ZoneInfo

from orion.cognition.workflows import next_run_for_recurring_schedule
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1


@dataclass
class ScheduledWorkflowEntry:
    request: WorkflowDispatchRequestV1
    next_run_utc: datetime


def _utc_now(now_utc: datetime | None = None) -> datetime:
    return (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def initial_next_run_utc(request: WorkflowDispatchRequestV1, *, now_utc: datetime | None = None) -> datetime | None:
    now = _utc_now(now_utc)
    schedule = request.execution_policy.schedule
    if schedule is None:
        return now
    if schedule.kind == "one_shot":
        if schedule.run_at_utc is None:
            return None
        return _coerce_utc(schedule.run_at_utc)
    return next_run_for_recurring_schedule(schedule=schedule, now_utc=now)


def register_schedule(
    *,
    schedules: Dict[str, ScheduledWorkflowEntry],
    request: WorkflowDispatchRequestV1,
    now_utc: datetime | None = None,
) -> Optional[ScheduledWorkflowEntry]:
    next_run = initial_next_run_utc(request, now_utc=now_utc)
    if next_run is None:
        return None
    entry = ScheduledWorkflowEntry(request=request, next_run_utc=next_run)
    schedules[request.request_id] = entry
    return entry


def due_schedules(
    schedules: Dict[str, ScheduledWorkflowEntry],
    *,
    now_utc: datetime | None = None,
) -> list[ScheduledWorkflowEntry]:
    now = _utc_now(now_utc)
    return [entry for entry in schedules.values() if entry.next_run_utc <= now]


def advance_after_dispatch(
    *,
    schedules: Dict[str, ScheduledWorkflowEntry],
    entry: ScheduledWorkflowEntry,
    now_utc: datetime | None = None,
) -> None:
    schedule = entry.request.execution_policy.schedule
    if schedule is None or schedule.kind == "one_shot":
        schedules.pop(entry.request.request_id, None)
        return
    next_run = next_run_for_recurring_schedule(schedule=schedule, now_utc=now_utc)
    if next_run is None:
        schedules.pop(entry.request.request_id, None)
        return
    entry.next_run_utc = next_run


def schedule_label(entry: ScheduledWorkflowEntry) -> str:
    schedule = entry.request.execution_policy.schedule
    if schedule is None:
        return "immediate"
    if schedule.kind == "one_shot" and schedule.run_at_utc is not None:
        return _coerce_utc(schedule.run_at_utc).isoformat()
    tz_name = schedule.timezone or "America/Denver"
    tz = ZoneInfo(tz_name)
    local = entry.next_run_utc.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M %Z")
