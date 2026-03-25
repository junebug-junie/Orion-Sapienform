from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List
from uuid import uuid4

from orion.cognition.workflows import next_run_for_recurring_schedule
from orion.schemas.workflow_execution import (
    WorkflowDispatchRequestV1,
    WorkflowScheduleAnalyticsV1,
    WorkflowScheduleEventRecordV1,
    WorkflowScheduleManageRequestV1,
    WorkflowScheduleManageResponseV1,
    WorkflowScheduleRecordV1,
    WorkflowScheduleRunRecordV1,
)


def _utc_now(now_utc: datetime | None = None) -> datetime:
    return (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)


@dataclass
class ClaimedSchedule:
    schedule: WorkflowScheduleRecordV1
    run: WorkflowScheduleRunRecordV1


@dataclass
class ScheduleAttentionSignal:
    schedule: WorkflowScheduleRecordV1
    analytics: WorkflowScheduleAnalyticsV1
    kind: str
    state: str
    transition: str


class WorkflowScheduleStore:
    def __init__(self, path: str, *, claim_ttl_seconds: int = 300, history_limit: int = 200) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._claim_ttl = max(30, int(claim_ttl_seconds))
        self._history_limit = max(20, int(history_limit))
        self._schedules: Dict[str, WorkflowScheduleRecordV1] = {}
        self._runs: List[WorkflowScheduleRunRecordV1] = []
        self._events: List[WorkflowScheduleEventRecordV1] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._persist()
            return
        raw = json.loads(self._path.read_text() or "{}")
        self._schedules = {
            str(item["schedule_id"]): WorkflowScheduleRecordV1.model_validate(item)
            for item in (raw.get("schedules") or [])
            if isinstance(item, dict)
        }
        self._runs = [WorkflowScheduleRunRecordV1.model_validate(item) for item in (raw.get("runs") or []) if isinstance(item, dict)]
        self._events = [WorkflowScheduleEventRecordV1.model_validate(item) for item in (raw.get("events") or []) if isinstance(item, dict)][-1000:]

    def _persist(self) -> None:
        data = {
            "schedules": [item.model_dump(mode="json") for item in self._schedules.values()],
            "runs": [item.model_dump(mode="json") for item in self._runs[-self._history_limit :]],
            "events": [item.model_dump(mode="json") for item in self._events[-1000:]],
        }
        temp = self._path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, sort_keys=True))
        temp.replace(self._path)

    def _event(self, *, kind: str, schedule_id: str, extra: dict[str, Any] | None = None) -> None:
        self._events.append(
            WorkflowScheduleEventRecordV1(
                event_id=str(uuid4()),
                kind=kind,
                schedule_id=schedule_id,
                occurred_at=_utc_now(),
                extra=dict(extra or {}),
            )
        )

    def upsert_from_dispatch(self, request: WorkflowDispatchRequestV1, *, now_utc: datetime | None = None) -> WorkflowScheduleRecordV1 | None:
        now = _utc_now(now_utc)
        schedule = request.execution_policy.schedule
        if schedule is None:
            return None
        if schedule.kind == "one_shot":
            next_run = schedule.run_at_utc
        else:
            next_run = next_run_for_recurring_schedule(schedule=schedule, now_utc=now)
        if next_run is None:
            return None
        with self._lock:
            existing = self._schedules.get(request.request_id)
            record = WorkflowScheduleRecordV1(
                schedule_id=(existing.schedule_id if existing else str(uuid4())),
                request_id=request.request_id,
                workflow_id=request.workflow_id,
                workflow_display_name=str(request.workflow_request.get("workflow_display_name") or request.workflow_id),
                workflow_request=dict(request.workflow_request or {}),
                execution_policy=request.execution_policy,
                notify_on=request.execution_policy.notify_on,
                source_service=request.source_service,
                source_kind=request.source_kind,
                source_correlation_id=request.correlation_id,
                created_at=(existing.created_at if existing else now),
                updated_at=now,
                next_run_at=next_run,
                last_run_at=(existing.last_run_at if existing else None),
                last_result_status=(existing.last_result_status if existing else "unknown"),
                state="scheduled",
                revision=(existing.revision + 1 if existing else 1),
                metadata=dict(existing.metadata if existing else {}),
            )
            self._schedules[record.schedule_id] = record
            if existing and existing.schedule_id != record.schedule_id:
                self._schedules.pop(existing.schedule_id, None)
            self._event(kind="schedule_created" if existing is None else "schedule_updated", schedule_id=record.schedule_id, extra={"workflow_id": record.workflow_id})
            self._persist()
            return record

    def list_schedules(self, *, include_inactive: bool = False) -> list[WorkflowScheduleRecordV1]:
        with self._lock:
            items = list(self._schedules.values())
            if not include_inactive:
                items = [item for item in items if item.state not in {"cancelled", "completed"}]
            return sorted(items, key=lambda item: (item.next_run_at or datetime.max.replace(tzinfo=timezone.utc), item.created_at))

    def _resolve_schedule(self, req: WorkflowScheduleManageRequestV1) -> tuple[WorkflowScheduleRecordV1 | None, list[WorkflowScheduleRecordV1]]:
        if req.schedule_id:
            item = self._schedules.get(req.schedule_id)
            return item, [] if item else []
        candidates = self.list_schedules(include_inactive=True)
        if req.workflow_id:
            candidates = [item for item in candidates if item.workflow_id == req.workflow_id]
        active = [item for item in candidates if item.state not in {"cancelled", "completed"}]
        if len(active) == 1:
            return active[0], []
        if len(active) > 1:
            return None, active
        if len(candidates) == 1:
            return candidates[0], []
        return None, candidates

    def _derive_analytics(self, schedule: WorkflowScheduleRecordV1, *, now_utc: datetime) -> WorkflowScheduleAnalyticsV1:
        runs = [run for run in self._runs if run.schedule_id == schedule.schedule_id]
        runs = sorted(runs, key=lambda run: run.dispatch_at, reverse=True)
        recent = runs[:5]
        success = [run for run in recent if str(run.status).lower() == "completed"]
        failures = [run for run in recent if str(run.status).lower() == "failed"]
        last_success = success[0].dispatch_at if success else None
        last_failure = failures[0].dispatch_at if failures else None
        outcomes = [str(run.status).lower() for run in recent]
        most_recent = outcomes[0] if outcomes else (schedule.last_result_status if schedule.last_result_status != "unknown" else None)

        is_overdue = bool(schedule.state == "scheduled" and schedule.next_run_at and schedule.next_run_at < now_utc)
        overdue_seconds = int((now_utc - schedule.next_run_at).total_seconds()) if is_overdue and schedule.next_run_at else None

        missed_run_count = 0
        spec = schedule.execution_policy.schedule
        if is_overdue and spec and spec.kind == "recurring":
            period_seconds = 0
            if spec.cadence == "daily":
                period_seconds = 86400
            elif spec.cadence == "weekly":
                period_seconds = 86400 * 7
            if period_seconds > 0 and overdue_seconds:
                missed_run_count = max(1, overdue_seconds // period_seconds)

        state = str(schedule.state).lower()
        if state == "paused":
            health = "paused"
        elif state == "cancelled":
            health = "cancelled"
        elif len(recent) == 0:
            health = "idle"
        elif (len(failures) >= 2 and len(success) == 0) or (is_overdue and len(success) == 0):
            health = "failing"
        elif len(failures) >= 1 or is_overdue:
            health = "degraded"
        else:
            health = "healthy"

        needs_attention = health in {"degraded", "failing"} or bool(is_overdue)
        return WorkflowScheduleAnalyticsV1(
            health=health,
            needs_attention=needs_attention,
            last_success_at=last_success,
            last_failure_at=last_failure,
            recent_run_count=len(recent),
            recent_success_count=len(success),
            recent_failure_count=len(failures),
            recent_outcomes=outcomes,
            most_recent_result_status=most_recent,
            is_overdue=is_overdue,
            overdue_seconds=overdue_seconds,
            missed_run_count=missed_run_count,
            history_window_runs=5,
        )

    def _with_analytics(self, schedule: WorkflowScheduleRecordV1, *, now_utc: datetime) -> WorkflowScheduleRecordV1:
        analytics = self._derive_analytics(schedule, now_utc=now_utc)
        return schedule.model_copy(update={"analytics": analytics}, deep=True)

    def evaluate_attention_signals(
        self,
        *,
        now_utc: datetime | None = None,
        overdue_min_seconds: int = 3600,
        reminder_cooldown_seconds: int = 21600,
    ) -> list[ScheduleAttentionSignal]:
        now = _utc_now(now_utc)
        signals: list[ScheduleAttentionSignal] = []
        min_overdue = max(0, int(overdue_min_seconds))
        reminder = max(0, int(reminder_cooldown_seconds))
        with self._lock:
            for schedule in self._schedules.values():
                spec = schedule.execution_policy.schedule
                if not spec or spec.kind != "recurring":
                    continue
                analytics = self._derive_analytics(schedule, now_utc=now)
                condition = "ok"
                if analytics.health == "failing":
                    condition = "failing"
                elif analytics.is_overdue and int(analytics.overdue_seconds or 0) >= min_overdue:
                    condition = "overdue"
                elif analytics.health == "degraded" and int(analytics.recent_failure_count or 0) >= 2:
                    condition = "degraded"

                state = "active" if condition != "ok" else "clear"
                attention = dict(schedule.metadata.get("attention") or {})
                previous_condition = str(attention.get("condition") or "ok")
                previous_state = str(attention.get("state") or "clear")
                last_notified_at_raw = attention.get("last_notified_at")
                last_notified_at = None
                if isinstance(last_notified_at_raw, str):
                    try:
                        last_notified_at = datetime.fromisoformat(last_notified_at_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
                    except Exception:
                        last_notified_at = None
                should_emit = False
                transition = "none"
                if state == "active":
                    if previous_state != "active" or previous_condition != condition:
                        should_emit = True
                        transition = "entered"
                    elif last_notified_at is None or (now - last_notified_at).total_seconds() >= reminder:
                        should_emit = True
                        transition = "reminder"
                elif previous_state == "active":
                    should_emit = True
                    transition = "recovered"

                attention.update(
                    {
                        "state": state,
                        "condition": condition,
                        "health": analytics.health,
                        "needs_attention": bool(analytics.needs_attention),
                        "updated_at": now.isoformat(),
                    }
                )
                if should_emit:
                    attention["last_notified_at"] = now.isoformat()
                    signals.append(
                        ScheduleAttentionSignal(
                            schedule=schedule.model_copy(deep=True),
                            analytics=analytics,
                            kind=condition,
                            state=state,
                            transition=transition,
                        )
                    )
                schedule.metadata["attention"] = attention
            if signals:
                self._persist()
        return signals

    def apply_management(self, req: WorkflowScheduleManageRequestV1, *, now_utc: datetime | None = None) -> WorkflowScheduleManageResponseV1:
        now = _utc_now(now_utc)
        with self._lock:
            if req.operation == "list":
                schedules = [self._with_analytics(item, now_utc=now) for item in self.list_schedules(include_inactive=True)]
                history: list[WorkflowScheduleRunRecordV1] = []
                if req.include_history:
                    history = self._runs[-20:]
                events = self._events[-20:] if req.include_history else []
                return WorkflowScheduleManageResponseV1(ok=True, operation=req.operation, request_id=req.request_id, message=f"{len(schedules)} schedule(s)", schedules=schedules, history=history, events=events)

            schedule, ambiguous = self._resolve_schedule(req)
            if schedule is None:
                err_code = "ambiguous_selection" if ambiguous else "schedule_not_found"
                return WorkflowScheduleManageResponseV1(
                    ok=False,
                    operation=req.operation,
                    request_id=req.request_id,
                    message="Ambiguous schedule selection." if ambiguous else "Schedule not found.",
                    schedules=ambiguous,
                    ambiguous=bool(ambiguous),
                    error_code=err_code,
                )

            if req.operation == "cancel":
                if schedule.state == "cancelled":
                    return WorkflowScheduleManageResponseV1(
                        ok=False,
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Schedule already cancelled.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="already_cancelled",
                    )
                schedule.state = "cancelled"
                schedule.updated_at = now
                schedule.revision += 1
                schedule.last_result_status = "cancelled"
                self._event(kind="schedule_cancelled", schedule_id=schedule.schedule_id)
            elif req.operation == "pause":
                if schedule.state == "paused":
                    return WorkflowScheduleManageResponseV1(
                        ok=False,
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Schedule already paused.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="already_paused",
                    )
                if schedule.state == "cancelled":
                    return WorkflowScheduleManageResponseV1(
                        ok=False,
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Cannot pause a cancelled schedule.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="unsupported_transition",
                    )
                schedule.state = "paused"
                schedule.updated_at = now
                schedule.revision += 1
                self._event(kind="schedule_paused", schedule_id=schedule.schedule_id)
            elif req.operation == "resume":
                if schedule.state == "cancelled":
                    return WorkflowScheduleManageResponseV1(
                        ok=False,
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Cannot resume a cancelled schedule.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="unsupported_transition",
                    )
                if schedule.state != "paused":
                    return WorkflowScheduleManageResponseV1(
                        ok=False,
                        operation=req.operation,
                        request_id=req.request_id,
                        message=f"Cannot resume schedule in state={schedule.state}.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="unsupported_transition",
                    )
                schedule.state = "scheduled"
                schedule.updated_at = now
                schedule.revision += 1
                self._event(kind="schedule_resumed", schedule_id=schedule.schedule_id)
            elif req.operation == "update":
                patch = req.patch
                if patch is None:
                    return WorkflowScheduleManageResponseV1(ok=False, operation=req.operation, request_id=req.request_id, message="Missing update patch.", error_code="missing_patch")
                if patch.expected_revision is not None and int(patch.expected_revision) != int(schedule.revision):
                    return WorkflowScheduleManageResponseV1(
                        ok=False,
                        operation=req.operation,
                        request_id=req.request_id,
                        message=f"Schedule revision conflict: expected {patch.expected_revision}, current {schedule.revision}.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="schedule_revision_conflict",
                        error_details={"expected_revision": int(patch.expected_revision), "current_revision": int(schedule.revision)},
                    )
                spec = schedule.execution_policy.schedule
                if spec is None:
                    return WorkflowScheduleManageResponseV1(ok=False, operation=req.operation, request_id=req.request_id, message="Schedule has no policy.", error_code="schedule_policy_missing")
                changed = spec.model_dump(mode="json")
                for field in ("run_at_utc", "cadence", "day_of_week", "hour_local", "minute_local", "timezone"):
                    value = getattr(patch, field)
                    if value is not None:
                        changed[field] = value
                try:
                    schedule.execution_policy.schedule = spec.model_validate(changed)
                except Exception as exc:
                    return WorkflowScheduleManageResponseV1(
                        ok=False,
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Invalid schedule update patch.",
                        error_code="invalid_patch",
                        error_details={"error": str(exc)},
                    )
                if patch.notify_on is not None:
                    schedule.execution_policy.notify_on = patch.notify_on
                    schedule.notify_on = patch.notify_on
                schedule.next_run_at = (
                    schedule.execution_policy.schedule.run_at_utc
                    if schedule.execution_policy.schedule.kind == "one_shot"
                    else next_run_for_recurring_schedule(schedule=schedule.execution_policy.schedule, now_utc=now)
                )
                schedule.updated_at = now
                schedule.revision += 1
                self._event(kind="schedule_updated", schedule_id=schedule.schedule_id)
            elif req.operation == "history":
                history = [run for run in self._runs if run.schedule_id == schedule.schedule_id][-20:]
                events = [event for event in self._events if event.schedule_id == schedule.schedule_id][-20:]
                return WorkflowScheduleManageResponseV1(ok=True, operation=req.operation, request_id=req.request_id, message=f"{len(history)} run(s)", schedule=self._with_analytics(schedule, now_utc=now), history=history, events=events)

            self._schedules[schedule.schedule_id] = schedule
            self._persist()
            return WorkflowScheduleManageResponseV1(ok=True, operation=req.operation, request_id=req.request_id, message=f"{req.operation} completed", schedule=self._with_analytics(schedule, now_utc=now))

    def claim_due(self, *, now_utc: datetime | None = None, limit: int = 10) -> list[ClaimedSchedule]:
        now = _utc_now(now_utc)
        claimed: list[ClaimedSchedule] = []
        with self._lock:
            candidates = [
                item
                for item in self._schedules.values()
                if item.state == "scheduled" and item.next_run_at is not None and item.next_run_at <= now
            ]
            candidates = sorted(candidates, key=lambda item: item.next_run_at or now)[: max(1, limit)]
            for schedule in candidates:
                run = WorkflowScheduleRunRecordV1(
                    run_id=str(uuid4()),
                    schedule_id=schedule.schedule_id,
                    workflow_id=schedule.workflow_id,
                    request_id=schedule.request_id,
                    status="dispatched",
                    dispatch_at=now,
                    metadata={
                        "notify_on": schedule.notify_on,
                        "claimed_for_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
                    },
                )
                schedule.last_run_at = now
                schedule.updated_at = now
                schedule.revision += 1
                schedule.last_result_status = "dispatched"
                recurring = schedule.execution_policy.schedule and schedule.execution_policy.schedule.kind == "recurring"
                if recurring:
                    schedule.next_run_at = next_run_for_recurring_schedule(schedule=schedule.execution_policy.schedule, now_utc=now)
                    schedule.state = "scheduled" if schedule.next_run_at else "completed"
                else:
                    schedule.next_run_at = None
                    schedule.state = "completed"
                self._runs.append(run)
                self._event(kind="schedule_due_claimed", schedule_id=schedule.schedule_id, extra={"run_id": run.run_id})
                self._event(kind="schedule_dispatched", schedule_id=schedule.schedule_id, extra={"run_id": run.run_id})
                claimed.append(ClaimedSchedule(schedule=schedule, run=run))
            if claimed:
                self._persist()
        return claimed

    def mark_dispatch_failed(self, *, run_id: str, schedule_id: str, error: str, now_utc: datetime | None = None) -> None:
        now = _utc_now(now_utc)
        with self._lock:
            for run in self._runs:
                if run.run_id == run_id:
                    run.status = "failed"
                    run.error = error
                    run.completed_at = now
                    break
            schedule = self._schedules.get(schedule_id)
            if schedule is not None:
                schedule.last_result_status = "failed"
                schedule.updated_at = now
                run = next((item for item in self._runs if item.run_id == run_id), None)
                claimed_for_raw = (run.metadata or {}).get("claimed_for_run_at") if run is not None else None
                claimed_for = None
                if isinstance(claimed_for_raw, str):
                    try:
                        claimed_for = datetime.fromisoformat(claimed_for_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
                    except Exception:
                        claimed_for = None
                if schedule.state == "completed" and schedule.execution_policy.schedule and schedule.execution_policy.schedule.kind == "recurring":
                    schedule.state = "scheduled"
                if schedule.execution_policy.schedule and schedule.execution_policy.schedule.kind == "recurring" and claimed_for is not None:
                    if schedule.next_run_at is None or claimed_for < schedule.next_run_at:
                        schedule.next_run_at = claimed_for
                    schedule.state = "scheduled"
                self._event(kind="schedule_run_failed", schedule_id=schedule_id, extra={"run_id": run_id, "error": error})
            self._persist()

    def mark_dispatch_succeeded(self, *, run_id: str, schedule_id: str, now_utc: datetime | None = None) -> None:
        now = _utc_now(now_utc)
        with self._lock:
            for run in self._runs:
                if run.run_id == run_id:
                    run.status = "completed"
                    run.completed_at = now
                    run.error = None
                    break
            schedule = self._schedules.get(schedule_id)
            if schedule is not None:
                schedule.last_result_status = "completed"
                schedule.updated_at = now
                spec = schedule.execution_policy.schedule
                if spec and spec.kind == "recurring":
                    schedule.next_run_at = next_run_for_recurring_schedule(schedule=spec, now_utc=now)
                    schedule.state = "scheduled" if schedule.next_run_at else "completed"
                self._event(kind="schedule_run_completed", schedule_id=schedule_id, extra={"run_id": run_id})
            self._persist()
