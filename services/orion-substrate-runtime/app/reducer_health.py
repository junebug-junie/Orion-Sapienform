"""In-process reducer health snapshots for /grammar/truth and operator diagnosis."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ReducerHealthClass = Literal[
    "healthy",
    "alive_behind",
    "dead_no_heartbeat",
    "blocked_on_event",
    "cursor_commit_failing",
    "reducer_disabled",
]

_LOCK = threading.Lock()
_SNAPSHOTS: dict[str, "ReducerHealthSnapshot"] = {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ReducerHealthSnapshot:
    reducer_key: str
    cursor_name: str
    enabled: bool = True
    last_tick_at: datetime | None = None
    last_success_at: datetime | None = None
    last_cursor_advance_at: datetime | None = None
    last_batch_events: int = 0
    last_error_at: datetime | None = None
    last_error_event_id: str | None = None
    last_error_reason: str | None = None
    blocked_event_id: str | None = None
    blocked_failures: int = 0
    quarantined_event_ids: list[str] = field(default_factory=list)
    unacknowledged_quarantine_count: int = 0
    pending_backlog: int | None = None
    stream_lag_sec: float | None = None
    cursor_wall_lag_sec: float | None = None

    def classify(
        self,
        *,
        heartbeat_stale_sec: float,
        stream_lag_degraded_sec: float,
    ) -> ReducerHealthClass:
        if not self.enabled:
            return "reducer_disabled"
        now = _utc_now()
        if self.last_tick_at is None:
            return "dead_no_heartbeat"
        heartbeat_age = (now - self.last_tick_at).total_seconds()
        if heartbeat_age > heartbeat_stale_sec:
            return "dead_no_heartbeat"
        if self.blocked_event_id and self.blocked_failures >= 1:
            return "blocked_on_event"
        if (
            self.last_success_at
            and self.last_cursor_advance_at
            and self.last_success_at > self.last_cursor_advance_at
        ):
            return "cursor_commit_failing"
        stream_lag = self.stream_lag_sec
        if stream_lag is not None and stream_lag > stream_lag_degraded_sec:
            return "alive_behind"
        return "healthy"

    def to_dict(
        self,
        *,
        heartbeat_stale_sec: float,
        stream_lag_degraded_sec: float,
    ) -> dict[str, Any]:
        classification = self.classify(
            heartbeat_stale_sec=heartbeat_stale_sec,
            stream_lag_degraded_sec=stream_lag_degraded_sec,
        )
        return {
            "reducer_key": self.reducer_key,
            "cursor_name": self.cursor_name,
            "enabled": self.enabled,
            "classification": classification,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_cursor_advance_at": (
                self.last_cursor_advance_at.isoformat() if self.last_cursor_advance_at else None
            ),
            "last_batch_events": self.last_batch_events,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            "last_error_event_id": self.last_error_event_id,
            "last_error_reason": self.last_error_reason,
            "blocked_event_id": self.blocked_event_id,
            "blocked_failures": self.blocked_failures,
            "quarantined_event_ids": list(self.quarantined_event_ids[-20:]),
            "unacknowledged_quarantine_count": self.unacknowledged_quarantine_count,
            "pending_backlog": self.pending_backlog,
            "stream_lag_sec": self.stream_lag_sec,
            "cursor_wall_lag_sec": self.cursor_wall_lag_sec,
        }


def _get(reducer_key: str, *, cursor_name: str, enabled: bool) -> ReducerHealthSnapshot:
    with _LOCK:
        snap = _SNAPSHOTS.get(reducer_key)
        if snap is None:
            snap = ReducerHealthSnapshot(
                reducer_key=reducer_key,
                cursor_name=cursor_name,
                enabled=enabled,
            )
            _SNAPSHOTS[reducer_key] = snap
        snap.enabled = enabled
        snap.cursor_name = cursor_name
        return snap


def record_tick(reducer_key: str, *, cursor_name: str, enabled: bool) -> None:
    snap = _get(reducer_key, cursor_name=cursor_name, enabled=enabled)
    with _LOCK:
        snap.last_tick_at = _utc_now()


def record_success(
    reducer_key: str,
    *,
    cursor_name: str,
    enabled: bool,
    batch_events: int,
) -> None:
    snap = _get(reducer_key, cursor_name=cursor_name, enabled=enabled)
    with _LOCK:
        now = _utc_now()
        snap.last_tick_at = now
        snap.last_success_at = now
        snap.last_batch_events = batch_events
        snap.blocked_event_id = None
        snap.blocked_failures = 0


def record_cursor_advance(reducer_key: str, *, cursor_name: str, enabled: bool) -> None:
    snap = _get(reducer_key, cursor_name=cursor_name, enabled=enabled)
    with _LOCK:
        snap.last_cursor_advance_at = _utc_now()


def record_error(
    reducer_key: str,
    *,
    cursor_name: str,
    enabled: bool,
    event_id: str | None,
    reason: str,
) -> None:
    snap = _get(reducer_key, cursor_name=cursor_name, enabled=enabled)
    with _LOCK:
        snap.last_tick_at = _utc_now()
        snap.last_error_at = _utc_now()
        snap.last_error_event_id = event_id
        snap.last_error_reason = reason
        if event_id:
            if snap.blocked_event_id == event_id:
                snap.blocked_failures += 1
            else:
                snap.blocked_event_id = event_id
                snap.blocked_failures = 1


def record_quarantine(
    reducer_key: str,
    *,
    cursor_name: str,
    enabled: bool,
    event_id: str,
) -> None:
    snap = _get(reducer_key, cursor_name=cursor_name, enabled=enabled)
    with _LOCK:
        if event_id not in snap.quarantined_event_ids:
            snap.quarantined_event_ids.append(event_id)
        snap.blocked_event_id = None
        snap.blocked_failures = 0


def update_quarantine_metrics(
    reducer_key: str,
    *,
    cursor_name: str,
    enabled: bool,
    unacknowledged_quarantine_count: int,
) -> None:
    snap = _get(reducer_key, cursor_name=cursor_name, enabled=enabled)
    with _LOCK:
        snap.unacknowledged_quarantine_count = unacknowledged_quarantine_count


def update_backlog_metrics(
    reducer_key: str,
    *,
    cursor_name: str,
    enabled: bool,
    pending_backlog: int,
    stream_lag_sec: float | None,
    cursor_wall_lag_sec: float | None,
) -> None:
    snap = _get(reducer_key, cursor_name=cursor_name, enabled=enabled)
    with _LOCK:
        snap.pending_backlog = pending_backlog
        snap.stream_lag_sec = stream_lag_sec
        snap.cursor_wall_lag_sec = cursor_wall_lag_sec


def health_snapshots() -> dict[str, dict[str, Any]]:
    with _LOCK:
        return {key: ReducerHealthSnapshot(**vars(snap)) for key, snap in _SNAPSHOTS.items()}


def clear_health_for_tests() -> None:
    with _LOCK:
        _SNAPSHOTS.clear()
