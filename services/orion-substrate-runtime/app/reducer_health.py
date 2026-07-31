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

# How recently `record_error()` must have fired for a success/advance timestamp
# inversion to count as a real commit failure. See `classify()` for the measured
# false-positive rate this exists to remove.
#
# 60s is deliberately generous relative to the 1s poll cadence: a reducer whose
# tick keeps raising re-records an error every second, clearing this bar by ~60x,
# while a batch that is merely in flight records nothing at all and can never
# reach it. Sized for the failure mode, not tuned to a measurement.
#
# Kept as a module constant rather than a parameter or an env key. It had a
# keyword parameter briefly; review pointed out `grammar_truth.py` is the only
# production caller and never passed it, so the parameter existed solely for two
# tests that tested the plumbing -- the same "knob with no operator use" smell as
# an env key, with an extra layer (CLAUDE.md section 0A, thin seams).
CURSOR_COMMIT_ERROR_GRACE_SEC = 60.0

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
        # `last_success_at > last_cursor_advance_at` on its own is NOT a failure
        # signal -- it is true by construction on every healthy batch, for as
        # long as everything between the two calls takes.
        #
        # `_process_events_with_poison_isolation()` calls `record_success()` the
        # moment `process_batch()` returns. The caller then does the rest of its
        # tick -- projection reload, `save_execution_trajectory`, `save_receipt`,
        # `_write_prediction_error_node` (a FalkorDB write) -- and only then
        # reaches `_advance_cursor()`. Those post-success writes dominate the
        # window, NOT the cursor commit itself: review measured the actual
        # SELECT+UPDATE at ~18ms (`cursor_positions[].updated_at` vs
        # `last_cursor_advance_at`), which against a multi-second batch interval
        # would produce a ~0.3% inversion rate, two orders below what is
        # observed. Anyone trying to shrink this window should look in
        # `worker.py`'s tick bodies, not at the commit.
        #
        # Measured live 2026-07-31 against a fully healthy substrate-runtime
        # (`last_error_at=None` on every sample, every reducer), two independent
        # runs against /grammar/truth:
        #
        #     execution_trajectory   32.5%  /  21.7%
        #     transport_bus            --   /  21.7%
        #     biometrics             22.5%  /  17.4%
        #     route_grammar          10.0%  /   8.7%
        #     chat_grammar             --   /   0.0%   (traffic-gated, idle)
        #
        # every inverted sample classified `cursor_commit_failing`. A
        # one-in-five-to-one-in-three false positive on the healthiest possible
        # state. That is the whole explanation for a CRITICAL page that has now
        # fired three times (2026-07-13 twice on biometrics_grammar_consumer,
        # 2026-07-31 on execution_grammar_reducer), each "self-resolving within
        # minutes with no reproducing evidence" -- there was never anything to
        # reproduce. The 15s recheck debounce added for the second fire
        # (SUBSTRATE_RUNTIME_HEALTH_RECHECK_DELAY_SEC) reduced the odds but
        # cannot fix a predicate that is wrong ~20% of the time.
        #
        # ## What actually reaches this branch
        #
        # NOT `_advance_cursor()`'s own two failure paths, despite the name.
        # Both pass a real `event_id` to `record_error()`, which sets
        # `blocked_event_id`/`blocked_failures`, and the `blocked_on_event`
        # branch above returns first. Verified by replaying the real call
        # sequence against this module:
        #
        #     record_success + record_error(event_id="gev_missing")  -> blocked_on_event
        #     record_success + record_error(event_id=None)           -> cursor_commit_failing
        #
        # So a genuinely stuck cursor commit pages as `reducer_blocked:<cursor>`,
        # and always has -- before this patch too. Detection is not lost here.
        #
        # What this branch genuinely covers is the `event_id=None` path: the poll
        # loops' own `record_error()` after a tick raised somewhere BETWEEN
        # `record_success()` and the advance -- a failed `publish_accepted_events`,
        # `save_execution_trajectory`, or `_write_prediction_error_node`. That is
        # a real condition and the reason this branch stays. `cursor_commit_failing`
        # is a misnomer for it; renaming is a contract change (the string reaches
        # `grammar_truth.py`'s degraded_reasons and the alert text) and is left
        # alone deliberately.
        #
        # Requiring a recent recorded error is therefore what separates "a tick
        # blew up mid-batch" from "the batch is simply still in flight".
        #
        # Deliberately NOT gated on `now - last_success_at` instead: under a real
        # failure the reducer keeps processing batches, so `last_success_at` keeps
        # refreshing and never looks stale.
        if (
            self.last_success_at
            and self.last_cursor_advance_at
            and self.last_success_at > self.last_cursor_advance_at
            and self.last_error_at is not None
            and (now - self.last_error_at).total_seconds()
            <= CURSOR_COMMIT_ERROR_GRACE_SEC
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
