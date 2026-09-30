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
from .workflow_schedule_metrics import WorkflowScheduleMetrics


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
    def __init__(
        self,
        path: str,
        *,
        claim_ttl_seconds: int = 300,
        history_limit: int = 200,
        metrics: WorkflowScheduleMetrics | None = None,
        max_dispatch_attempts: int = 3,
        retry_backoff_seconds: int = 300,
    ) -> None:
        self._path = self._resolve_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._claim_ttl = max(30, int(claim_ttl_seconds))
        self._history_limit = max(20, int(history_limit))
        self._max_dispatch_attempts = max(1, int(max_dispatch_attempts))
        self._retry_backoff_seconds = max(0, int(retry_backoff_seconds))
        self._schedules: Dict[str, WorkflowScheduleRecordV1] = {}
        self._runs: List[WorkflowScheduleRunRecordV1] = []
        self._events: List[WorkflowScheduleEventRecordV1] = []
        self._metrics = metrics
        # Terminal rows of durable runs no schedule run was (yet) waiting on: run_id -> (status,
        # error). A run can end before the scheduler loop records the accepted reply (an instant
        # pool refusal, or an in-flight run a re-dispatch found finishing in the gap);
        # mark_awaiting_durable settles from here instead of waiting ~18h for the reaper. Bounded,
        # in memory: the reaper + deterministic re-dispatch still cover a restart in between.
        self._early_terminals: Dict[str, tuple[str, str | None]] = {}
        self._load()

    def _error_response(
        self,
        *,
        operation: str,
        request_id: str | None,
        message: str,
        error_code: str,
        **kwargs: Any,
    ) -> WorkflowScheduleManageResponseV1:
        if self._metrics is not None:
            self._metrics.incr_error(error_code)
        return WorkflowScheduleManageResponseV1(
            ok=False,
            operation=operation,
            request_id=request_id,
            message=message,
            error_code=error_code,
            **kwargs,
        )

    @staticmethod
    def _resolve_path(path: str) -> Path:
        candidate = Path(path or "").expanduser()
        if not str(candidate).strip():
            # Intentionally /tmp -- see the comment on settings.py's
            # actions_workflow_schedule_store_path default for why.
            candidate = Path("/tmp/orion-actions/workflow_schedules.json")
        if candidate.exists() and candidate.is_dir():
            return candidate / "workflow_schedules.json"
        if str(path).endswith("/"):
            return candidate / "workflow_schedules.json"
        return candidate

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
            # _schedules is keyed by schedule_id; the dispatch upsert key is the
            # caller's request_id, so match on the record field.
            existing = next(
                (item for item in self._schedules.values() if item.request_id == request.request_id),
                None,
            )
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
        elif len(failures) >= 1 or is_overdue or len(success) == 0:
            # len(success) == 0 here means every recent run is still sitting at
            # "dispatched" (or some other non-terminal status) -- no confirmed
            # success and no confirmed failure. That's not the same as healthy.
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
                elif analytics.health == "degraded" and self._consecutive_failures(schedule) >= self._max_dispatch_attempts:
                    # `health` summarises the last 5 runs, so it stays "degraded"
                    # for days after a schedule has recovered. Paging off that
                    # alone kept re-firing this signal every reminder-cooldown
                    # window on a schedule whose runs were all succeeding --
                    # confirmed live 2026-08-21..09-01: 8 notifications/day (this
                    # signal is published twice, generic + pending-attention) for
                    # github_compactor_pass, including every day it completed. The
                    # old `recent_failure_count >= 2` guard could not catch that,
                    # because failures arrive in retry bursts and a burst is
                    # always >= 2.
                    #
                    # The condition is now the retry budget running out: the point
                    # at which the store has stopped retrying on its own and a
                    # human is the only thing that can move it. That is also why
                    # this does not page on a transient blip that the very next
                    # retry fixes, and why a success clears it.
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
                return self._error_response(
                    operation=req.operation,
                    request_id=req.request_id,
                    message="Ambiguous schedule selection." if ambiguous else "Schedule not found.",
                    schedules=ambiguous,
                    ambiguous=bool(ambiguous),
                    error_code=err_code,
                )

            if req.operation == "cancel":
                if schedule.state == "cancelled":
                    return self._error_response(
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
                    return self._error_response(
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Schedule already paused.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="already_paused",
                    )
                if schedule.state == "cancelled":
                    return self._error_response(
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
                    return self._error_response(
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Cannot resume a cancelled schedule.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="unsupported_transition",
                    )
                if schedule.state != "paused":
                    return self._error_response(
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
                    return self._error_response(operation=req.operation, request_id=req.request_id, message="Missing update patch.", error_code="missing_patch")
                if patch.expected_revision is not None and int(patch.expected_revision) != int(schedule.revision):
                    return self._error_response(
                        operation=req.operation,
                        request_id=req.request_id,
                        message=f"Schedule revision conflict: expected {patch.expected_revision}, current {schedule.revision}.",
                        schedule=self._with_analytics(schedule, now_utc=now),
                        error_code="schedule_revision_conflict",
                        error_details={"expected_revision": int(patch.expected_revision), "current_revision": int(schedule.revision)},
                    )
                spec = schedule.execution_policy.schedule
                if spec is None:
                    return self._error_response(operation=req.operation, request_id=req.request_id, message="Schedule has no policy.", error_code="schedule_policy_missing")
                changed = spec.model_dump(mode="json")
                for field in ("run_at_utc", "cadence", "day_of_week", "hour_local", "minute_local", "timezone"):
                    value = getattr(patch, field)
                    if value is not None:
                        changed[field] = value
                try:
                    schedule.execution_policy.schedule = spec.model_validate(changed)
                except Exception as exc:
                    return self._error_response(
                        operation=req.operation,
                        request_id=req.request_id,
                        message="Invalid schedule update patch.",
                        error_code="invalid_patch",
                        error_details={"error": str(exc)},
                    )
                if patch.notify_on is not None:
                    # Must go through _apply_notify_on: writing only the record and
                    # policy fields (as this did) left the authoritative embedded
                    # copy stale, so an operator changing notify_on through the
                    # management API saw no change in what actually notified them.
                    self._apply_notify_on(schedule, patch.notify_on)
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
            self._reap_stale_claims(now=now)
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

    @staticmethod
    def _consecutive_failures(schedule: WorkflowScheduleRecordV1) -> int:
        """How many dispatches in a row have failed since the last success.

        Persisted on the record rather than derived from run history. History is
        truncated to `_history_limit` on every save and the run list is global
        across schedules, so a derived count answers differently before and after
        a restart -- always in the "grant more retries" direction, and the live
        store is already sitting exactly at the cap with one schedule owning most
        of the window. That is the same shape as the restart-resets-the-daily-cap
        incident, so the bound gets durable state instead of a lossy derivation.
        """
        try:
            return max(0, int((schedule.metadata or {}).get("consecutive_failures") or 0))
        except (TypeError, ValueError):
            return 0

    def _apply_notify_on(self, schedule: WorkflowScheduleRecordV1, notify_on: str) -> bool:
        """Write notify_on to all three copies. Caller holds the lock and persists.

        The copies are not interchangeable: the record field, the record's
        execution_policy, and the execution_policy embedded in `workflow_request`.
        The last one is authoritative -- `_dispatch_scheduled_workflow` forwards
        `workflow_request["execution_policy"]` to cortex-orch, which validates it
        into the policy `_emit_workflow_notify` reads -- so a caller that updates
        only the record field changes nothing a user can observe.

        Returns True when a change was written.
        """
        request = dict(schedule.workflow_request or {})
        embedded = dict(request.get("execution_policy") or {})
        unchanged = (
            schedule.notify_on == notify_on
            and schedule.execution_policy.notify_on == notify_on
            # An absent embedded policy has nothing to disagree with; treating it
            # as a mismatch would rewrite and re-persist the record on every boot.
            and (not embedded or embedded.get("notify_on") == notify_on)
        )
        if unchanged:
            return False
        previous = schedule.notify_on
        schedule.notify_on = notify_on
        schedule.execution_policy = schedule.execution_policy.model_copy(update={"notify_on": notify_on})
        if embedded:
            embedded["notify_on"] = notify_on
            request["execution_policy"] = embedded
            schedule.workflow_request = request
        self._event(
            kind="schedule_notify_on_updated",
            schedule_id=schedule.schedule_id,
            extra={"from": previous, "to": notify_on},
        )
        return True

    def _reap_stale_claims(self, *, now: datetime) -> None:
        """Fail any dispatch still in flight past the claim TTL. Caller holds the lock.

        `_claim_ttl` was accepted by the constructor and never read anywhere in the
        service, so nothing ever reaped a hung dispatch: the live store still holds
        an orphaned `dispatched` row from 2026-08-20T16:24:52Z, 13 days later. That
        orphan is not just untidy -- it is the newest run for its schedule, so it
        makes `most_recent_result_status` read "dispatched" and silences the
        attention signal on a schedule that is genuinely stuck. Reaping routes the
        orphan through the normal failure path, which is also the correct handling:
        a hung RPC is exactly what a bounded retry is for.
        """
        ttl = timedelta(seconds=self._claim_ttl)
        for run in list(self._runs):
            if str(run.status).lower() != "dispatched":
                continue
            awaiting_until = self._awaiting_until(run)
            if awaiting_until is not None:
                # Handed to a durable run (compactor.digest): its terminal state row settles this
                # run (settle_durable_run). Only if none arrived by the run's own deadline (plus a
                # margin) is it failed here -- the claim TTL covers the synchronous dispatch only.
                if awaiting_until > now:
                    continue
                self._mark_failed_locked(
                    run_id=run.run_id,
                    schedule_id=run.schedule_id,
                    error=f"durable_run_completion_unobserved:{(run.metadata or {}).get('durable_run_id')}",
                    now=now,
                )
                continue
            if run.dispatch_at + ttl > now:
                continue
            self._mark_failed_locked(
                run_id=run.run_id,
                schedule_id=run.schedule_id,
                error=f"claim_expired_after_{self._claim_ttl}s",
                now=now,
            )

    def set_notify_on(self, *, schedule_id: str, notify_on: str, now_utc: datetime | None = None) -> bool:
        """Public entry point for changing a schedule's notification policy."""
        now = _utc_now(now_utc)
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if schedule is None:
                return False
            if not self._apply_notify_on(schedule, notify_on):
                return False
            schedule.updated_at = now
            schedule.revision += 1
            self._persist()
            return True

    def mark_dispatch_failed(self, *, run_id: str, schedule_id: str, error: str, now_utc: datetime | None = None) -> None:
        now = _utc_now(now_utc)
        with self._lock:
            self._mark_failed_locked(run_id=run_id, schedule_id=schedule_id, error=error, now=now)
            self._persist()

    def _mark_failed_locked(self, *, run_id: str, schedule_id: str, error: str, now: datetime) -> None:
        """Record one failed dispatch and decide whether to retry it.

        Caller holds the lock and is responsible for persisting.
        """
        run = next((item for item in self._runs if item.run_id == run_id), None)
        if run is not None:
            run.status = "failed"
            run.error = error
            run.completed_at = now
        schedule = self._schedules.get(schedule_id)
        if schedule is None:
            return
        # An outcome for a *superseded* claim updates the run, not the schedule.
        # A later claim has already been made and has already reported, so this
        # one's due-slot is history: re-arming it would run the workflow off its
        # own schedule, and folding it into the failure budget would let ancient
        # orphans page about a schedule that is currently fine. Observed live on
        # deploy 2026-09-02: reaping the 2026-08-20 orphan armed a retry that
        # dispatched a real compactor run at 05:40 UTC, 6.5h off its 12:10 slot,
        # for a due-slot the job had already passed 12 times.
        if (
            run is not None
            and schedule.last_run_at is not None
            and run.dispatch_at < schedule.last_run_at
        ):
            self._event(
                kind="schedule_run_failed_superseded",
                schedule_id=schedule_id,
                extra={
                    "run_id": run_id,
                    "error": error,
                    "dispatch_at": run.dispatch_at.isoformat(),
                    "superseded_by_run_at": schedule.last_run_at.isoformat(),
                },
            )
            return
        schedule.last_result_status = "failed"
        schedule.updated_at = now
        attempts = self._consecutive_failures(schedule) + 1
        metadata = dict(schedule.metadata or {})
        metadata["consecutive_failures"] = attempts
        schedule.metadata = metadata

        spec = schedule.execution_policy.schedule
        recurring = bool(spec and spec.kind == "recurring")
        if schedule.state == "completed" and recurring:
            schedule.state = "scheduled"
        # A schedule an operator cancelled or paused must not be revived by a
        # failure -- least of all armed to run again one backoff from now. The
        # in-flight dispatch whose failure lands here may have been claimed
        # before the cancel.
        if recurring and schedule.state not in {"cancelled", "paused"}:
            # Retry the same slot, but bounded and with backoff. Rewinding
            # next_run_at straight to the claimed slot (a time already in the
            # past) made the very next scheduler tick re-claim it, so a
            # persistently failing workflow retried at poll cadence forever:
            # confirmed live 2026-08-20, 343 failure notifications from one
            # schedule in a single day, each retry a full LLM workflow run.
            # The loop was only ever broken by a dispatch that hung without
            # being marked, not by any guard.
            if attempts >= self._max_dispatch_attempts:
                # Budget spent: leave next_run_at on the next natural occurrence
                # (claim_due already advanced it) and let the attention signal
                # page instead of burning more runs.
                self._event(
                    kind="schedule_retry_budget_exhausted",
                    schedule_id=schedule_id,
                    extra={
                        "run_id": run_id,
                        "attempts": attempts,
                        "next_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
                    },
                )
            else:
                retry_at = now + timedelta(seconds=self._retry_backoff_seconds * (2 ** (attempts - 1)))
                # Only ever move the run earlier. A backoff longer than the gap to
                # the next natural occurrence must not push a real run later.
                applied = schedule.next_run_at is None or retry_at < schedule.next_run_at
                if applied:
                    schedule.next_run_at = retry_at
                self._event(
                    kind="schedule_retry_scheduled",
                    schedule_id=schedule_id,
                    extra={
                        "run_id": run_id,
                        "attempts": attempts,
                        # The time actually computed, and whether it was taken --
                        # reporting next_run_at unconditionally made this event
                        # claim a retry was scheduled when the guard declined it.
                        "retry_at": retry_at.isoformat(),
                        "applied": applied,
                        "next_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
                    },
                )
            schedule.state = "scheduled"
        self._event(kind="schedule_run_failed", schedule_id=schedule_id, extra={"run_id": run_id, "error": error})

    def mark_dispatch_succeeded(self, *, run_id: str, schedule_id: str, now_utc: datetime | None = None) -> None:
        now = _utc_now(now_utc)
        with self._lock:
            self._mark_succeeded_locked(run_id=run_id, schedule_id=schedule_id, now=now)
            self._persist()

    def _mark_succeeded_locked(self, *, run_id: str, schedule_id: str, now: datetime) -> None:
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
            if (schedule.metadata or {}).get("consecutive_failures"):
                metadata = dict(schedule.metadata or {})
                metadata.pop("consecutive_failures", None)
                schedule.metadata = metadata
            spec = schedule.execution_policy.schedule
            if spec and spec.kind == "recurring":
                schedule.next_run_at = next_run_for_recurring_schedule(schedule=spec, now_utc=now)
                schedule.state = "scheduled" if schedule.next_run_at else "completed"
            self._event(kind="schedule_run_completed", schedule_id=schedule_id, extra={"run_id": run_id})

    # --- durable completion (compactor.digest) --------------------------------------------------

    @staticmethod
    def _awaiting_until(run: WorkflowScheduleRunRecordV1) -> datetime | None:
        raw = (run.metadata or {}).get("awaiting_until")
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def mark_awaiting_durable(
        self,
        *,
        run_id: str,
        schedule_id: str,
        durable_run_id: str,
        awaiting_until: datetime,
        now_utc: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """The dispatch was accepted as a durable run: the run stays ``dispatched`` (in flight --
        attention stays quiet, no retry is armed) until that run's terminal state row arrives
        (``settle_durable_run``) or ``awaiting_until`` passes (the reaper fails it). Returns the
        settlements made at once when that row already arrived (see ``settle_durable_run``)."""
        now = _utc_now(now_utc)
        with self._lock:
            run = next((item for item in self._runs if item.run_id == run_id), None)
            if run is None:
                return []
            run.metadata = {
                **dict(run.metadata or {}),
                "durable_run_id": durable_run_id,
                "awaiting_until": awaiting_until.astimezone(timezone.utc).isoformat(),
                "accepted_at": now.isoformat(),
            }
            self._event(
                kind="schedule_run_awaiting_durable",
                schedule_id=schedule_id,
                extra={"run_id": run_id, "durable_run_id": durable_run_id,
                       "awaiting_until": run.metadata["awaiting_until"]},
            )
            early = self._early_terminals.pop(durable_run_id, None)
            if early is not None:
                # It already ended: settle now through the same path a late row would take.
                settled = self._settle_locked(durable_run_id=durable_run_id, status=early[0], error=early[1], now=now)
                self._persist()
                return settled
            self._persist()
            return []

    def settle_durable_run(
        self,
        *,
        durable_run_id: str,
        status: str,
        error: str | None = None,
        now_utc: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """A durable run a schedule run is waiting on ended (``completed`` / ``failed`` /
        ``cancelled``). Settles every still-``dispatched`` schedule run waiting on it through the
        normal success/failure paths (retry budget, attention, next occurrence). A completion for
        a superseded claim updates the run only, like a superseded failure. Returns one dict per
        settled schedule run (run_id, schedule_id, workflow_id, notify_on, recipient_group, status,
        error). Nothing waiting: the row is remembered for ``mark_awaiting_durable`` (bounded) and
        [] is returned -- a replayed row for an already-settled run is simply ignored there."""
        now = _utc_now(now_utc)
        with self._lock:
            settled = self._settle_locked(durable_run_id=durable_run_id, status=status, error=error, now=now)
            if settled:
                self._persist()
            elif not any((r.metadata or {}).get("durable_run_id") == durable_run_id for r in self._runs):
                self._early_terminals[durable_run_id] = (status, error)
                while len(self._early_terminals) > 200:
                    self._early_terminals.pop(next(iter(self._early_terminals)))
        return settled

    def _settle_locked(self, *, durable_run_id: str, status: str, error: str | None, now: datetime) -> list[dict[str, Any]]:
        settled: list[dict[str, Any]] = []
        for run in list(self._runs):
            if str(run.status).lower() != "dispatched":
                continue
            if (run.metadata or {}).get("durable_run_id") != durable_run_id:
                continue
            schedule = self._schedules.get(run.schedule_id)
            if status == "completed":
                if schedule is not None and schedule.last_run_at is not None and run.dispatch_at < schedule.last_run_at:
                    run.status, run.completed_at, run.error = "completed", now, None
                    self._event(kind="schedule_run_completed_superseded", schedule_id=run.schedule_id,
                                extra={"run_id": run.run_id, "durable_run_id": durable_run_id})
                else:
                    self._mark_succeeded_locked(run_id=run.run_id, schedule_id=run.schedule_id, now=now)
            else:
                detail = f":{error}" if error else ""
                self._mark_failed_locked(run_id=run.run_id, schedule_id=run.schedule_id,
                                         error=f"durable_run_{status}{detail}"[:500], now=now)
            settled.append({
                "run_id": run.run_id,
                "schedule_id": run.schedule_id,
                "workflow_id": run.workflow_id,
                "notify_on": schedule.notify_on if schedule is not None else None,
                "recipient_group": schedule.execution_policy.recipient_group if schedule is not None else None,
                "status": status,
                "error": error,
                "durable_run_id": durable_run_id,
            })
        return settled
