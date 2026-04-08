from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from orion.schemas.workflow_execution import (
    WorkflowExecutionPolicyV1,
    WorkflowNotifyOn,
    WorkflowScheduleSpecV1,
)

_DAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", str(prompt or "").strip().lower())


def _hour_from_token(token: str) -> Optional[int]:
    raw = str(token or "").strip().lower()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    if value < 0:
        return None
    if value <= 11:
        return value
    if value <= 23:
        return value
    return None


def _parse_time_for_schedule(normalized_prompt: str) -> tuple[int, int] | None:
    m = re.search(r"(?:for|at)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", normalized_prompt)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    meridiem = (m.group(3) or "").lower()
    if hour > 12 or minute > 59:
        return None
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return hour, minute


def _has_explicit_schedule_intent(normalized_prompt: str) -> bool:
    return any(token in normalized_prompt for token in ("schedule ", "scheduled ", "every ", "tomorrow", "tonight", "in one minute"))


def _parse_notify_on(normalized_prompt: str) -> WorkflowNotifyOn:
    if any(phrase in normalized_prompt for phrase in ("only on failure", "if it fails", "if it fail", "on failure")):
        return "failure"
    if any(phrase in normalized_prompt for phrase in ("notify me when done", "message me when it", "let me know when it", "notify me when it", "when done")):
        return "completion"
    return "none"


def _next_weekday_run(now_local: datetime, *, weekday: int, hour_local: int, minute_local: int) -> datetime:
    day_delta = (weekday - now_local.weekday()) % 7
    candidate = now_local + timedelta(days=day_delta)
    candidate = candidate.replace(hour=hour_local, minute=minute_local, second=0, microsecond=0)
    if candidate <= now_local:
        candidate = candidate + timedelta(days=7)
    return candidate


def _one_shot_tonight(now_local: datetime, *, hour_local: int) -> datetime:
    candidate = now_local.replace(hour=hour_local, minute=0, second=0, microsecond=0)
    if candidate <= now_local:
        candidate = candidate + timedelta(days=1)
    return candidate


def derive_workflow_execution_policy(
    *,
    workflow_id: str,
    prompt: str,
    session_id: str | None,
    user_id: str | None,
    recipient_group: str = "juniper_primary",
    default_timezone: str = "America/Denver",
    now_utc: datetime | None = None,
) -> WorkflowExecutionPolicyV1:
    normalized = _normalize_prompt(prompt)
    tz = ZoneInfo(default_timezone)
    now = (now_utc or datetime.now(timezone.utc)).astimezone(tz)

    notify_on = _parse_notify_on(normalized)
    schedule: WorkflowScheduleSpecV1 | None = None
    summary = "Runs immediately."

    explicit_schedule_intent = _has_explicit_schedule_intent(normalized)

    if "tomorrow morning" in normalized and explicit_schedule_intent:
        run_local = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        schedule = WorkflowScheduleSpecV1(
            kind="one_shot",
            timezone=default_timezone,
            run_at_utc=run_local.astimezone(timezone.utc),
            label="tomorrow morning",
        )
        summary = f"Scheduled one-shot for {run_local.strftime('%Y-%m-%d %H:%M %Z')}."
    elif "in one minute" in normalized and ("run " in normalized or "schedule " in normalized):
        run_local = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        schedule = WorkflowScheduleSpecV1(
            kind="one_shot",
            timezone=default_timezone,
            run_at_utc=run_local.astimezone(timezone.utc),
            label="in one minute",
        )
        summary = f"Scheduled one-shot for {run_local.strftime('%Y-%m-%d %H:%M %Z')}."
    else:
        nightly = re.search(r"every night(?: at (\d{1,2}))?", normalized)
        weekly = re.search(r"every (monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?: at (\d{1,2}))?", normalized)
        tonight = re.search(r"tonight at (\d{1,2})", normalized)
        explicit_time = _parse_time_for_schedule(normalized) if explicit_schedule_intent else None
        if explicit_time is not None:
            hour, minute = explicit_time
            run_local = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if run_local <= now:
                run_local = run_local + timedelta(days=1)
            schedule = WorkflowScheduleSpecV1(
                kind="one_shot",
                timezone=default_timezone,
                run_at_utc=run_local.astimezone(timezone.utc),
                label=f"for {hour:02d}:{minute:02d}",
            )
            summary = f"Scheduled one-shot for {run_local.strftime('%Y-%m-%d %H:%M %Z')}."
        elif nightly:
            hour = _hour_from_token(nightly.group(1) or "23") or 23
            schedule = WorkflowScheduleSpecV1(
                kind="recurring",
                cadence="daily",
                timezone=default_timezone,
                hour_local=hour,
                minute_local=0,
                label=f"every night at {hour:02d}:00",
            )
            summary = f"Scheduled recurring daily run at {hour:02d}:00 {default_timezone}."
        elif weekly:
            day = (weekly.group(1) or "").lower()
            hour = _hour_from_token(weekly.group(2) or "9") or 9
            day_idx = _DAY_INDEX[day]
            schedule = WorkflowScheduleSpecV1(
                kind="recurring",
                cadence="weekly",
                timezone=default_timezone,
                day_of_week=day_idx,
                hour_local=hour,
                minute_local=0,
                label=f"every {day} at {hour:02d}:00",
            )
            summary = f"Scheduled recurring weekly run every {day.title()} at {hour:02d}:00 {default_timezone}."
        elif tonight:
            hour = _hour_from_token(tonight.group(1)) or 2
            run_local = _one_shot_tonight(now, hour_local=hour)
            schedule = WorkflowScheduleSpecV1(
                kind="one_shot",
                timezone=default_timezone,
                run_at_utc=run_local.astimezone(timezone.utc),
                label=f"tonight at {hour:02d}:00",
            )
            summary = f"Scheduled one-shot for {run_local.strftime('%Y-%m-%d %H:%M %Z')}."

    invocation_mode = "scheduled" if schedule is not None else "immediate"
    if invocation_mode == "immediate" and notify_on != "none":
        summary = "Runs immediately with notification enabled."
    elif invocation_mode == "scheduled" and notify_on != "none":
        summary = f"{summary} Notification policy: {notify_on}."

    return WorkflowExecutionPolicyV1(
        workflow_id=workflow_id,
        invocation_mode=invocation_mode,
        schedule=schedule,
        notify_on=notify_on,
        recipient_group=recipient_group,
        session_id=session_id,
        origin_user_id=user_id,
        policy_summary=summary,
        requested_from_chat=True,
    )


def next_run_for_recurring_schedule(*, schedule: WorkflowScheduleSpecV1, now_utc: datetime | None = None) -> datetime | None:
    if schedule.kind != "recurring":
        return None
    tz = ZoneInfo(schedule.timezone or "America/Denver")
    now = (now_utc or datetime.now(timezone.utc)).astimezone(tz)
    hour = int(schedule.hour_local or 0)
    minute = int(schedule.minute_local or 0)
    if schedule.cadence == "daily":
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        return candidate.astimezone(timezone.utc)
    if schedule.cadence == "weekly" and schedule.day_of_week is not None:
        return _next_weekday_run(now, weekday=int(schedule.day_of_week), hour_local=hour, minute_local=minute).astimezone(timezone.utc)
    return None
