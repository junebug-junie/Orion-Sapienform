"""`cursor_commit_failing` must require a real error, not timestamp ordering.

Third-fire regression guard. `reducer_cursor_commit_failing:*` paged CRITICAL on
2026-07-13 (twice, `biometrics_grammar_consumer`) and 2026-07-31 (once,
`execution_grammar_reducer`). All three "self-resolved within minutes with no
reproducing evidence", and the second investigation
(`docs/superpowers/pr-reports/2026-07-13-substrate-health-recheck-debounce-pr.md`)
explicitly wrote: "If this alert fires a third time, that would be strong evidence
of a real, recurring issue worth a deeper dive."

It was real, and it was in the detector.

`_process_events_with_poison_isolation()` calls `record_success()` as soon as
`process_batch()` returns; the caller only then runs `_advance_cursor()`, which
does a Postgres SELECT plus an UPDATE on a worker thread. For that whole window
`last_success_at > last_cursor_advance_at` is true on a perfectly healthy
reducer, and the old predicate called that a commit failure.

Measured live 2026-07-31 against a healthy substrate-runtime (every reducer had
`last_error_at=None`, `blocked_failures=0`), sampling /grammar/truth every 200ms
for 90s:

    execution_trajectory   inverted in 13/40 samples (32.5%)
    biometrics             inverted in  9/40 samples (22.5%)
    route_grammar          inverted in  4/40 samples (10.0%)

all classified `cursor_commit_failing`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reducer_health import (  # noqa: E402
    DEFAULT_CURSOR_COMMIT_ERROR_GRACE_SEC,
    ReducerHealthSnapshot,
)

HEARTBEAT_STALE_SEC = 120.0
STREAM_LAG_DEGRADED_SEC = 60.0


# The previous batch's cursor advance. Any value comfortably older than the
# in-flight window works; execution batches land roughly every 20s live.
PREVIOUS_ADVANCE_SEC = 20.0


def _snap(*, success_offset: float, advance_offset: float, error_offset: float | None):
    """Build a snapshot with timestamps expressed as seconds-ago.

    Direction matters and is easy to get backwards (this helper was written
    inverted the first time, caught by the explicit precondition assert below).
    "In flight" means `record_success()` has just fired for the CURRENT batch
    while `last_cursor_advance_at` still holds the PREVIOUS batch's time --
    so a small `success_offset` against a large `advance_offset`.
    """
    now = datetime.now(timezone.utc)
    return ReducerHealthSnapshot(
        reducer_key="execution_trajectory",
        cursor_name="execution_grammar_reducer",
        enabled=True,
        last_tick_at=now,
        last_success_at=now - timedelta(seconds=success_offset),
        last_cursor_advance_at=now - timedelta(seconds=advance_offset),
        last_error_at=None if error_offset is None else now - timedelta(seconds=error_offset),
    )


def _classify(snap, **kw):
    return snap.classify(
        heartbeat_stale_sec=HEARTBEAT_STALE_SEC,
        stream_lag_degraded_sec=STREAM_LAG_DEGRADED_SEC,
        **kw,
    )


class TestInFlightCommitIsNotAFailure:
    def test_the_exact_live_window_is_healthy(self):
        """The real numbers off the live endpoint, 2026-07-31 20:28:5x.

        last_success_at   20:28:51.632147
        last_cursor_advance_at 20:28:52.043960   -> 411ms later

        Inverted for 411ms, no error recorded. This is the state that paged.
        """
        snap = _snap(
            success_offset=0.0,
            advance_offset=0.411 + PREVIOUS_ADVANCE_SEC,
            error_offset=None,
        )
        assert snap.last_success_at > snap.last_cursor_advance_at, "precondition: inverted"
        assert _classify(snap) == "healthy"

    @pytest.mark.parametrize("window_sec", [0.021, 0.411, 1.0, 5.0])
    def test_any_inversion_without_an_error_is_healthy(self, window_sec):
        """Width of the window is irrelevant when nothing failed.

        21ms is the measured biometrics window, 411ms the execution one. Even a
        multi-second commit is just a slow commit, not a failing one -- the
        distinguishing fact is whether `record_error()` ever fired.
        """
        snap = _snap(
            success_offset=0.0,
            advance_offset=window_sec + PREVIOUS_ADVANCE_SEC,
            error_offset=None,
        )
        assert snap.last_success_at > snap.last_cursor_advance_at, "precondition: inverted"
        assert _classify(snap) == "healthy"


class TestRealCommitFailureStillDetected:
    def test_inversion_plus_recent_error_is_still_a_failure(self):
        """The genuine case must not be silenced by this fix.

        `_advance_cursor()` calls `record_error()` on both of its failure paths,
        at the 1s poll cadence, so a real stuck commit always has a fresh error.
        """
        snap = _snap(success_offset=0.0, advance_offset=30.0, error_offset=0.5)
        assert _classify(snap) == "cursor_commit_failing"

    def test_error_just_inside_the_grace_window_still_fires(self):
        snap = _snap(
            success_offset=0.0,
            advance_offset=300.0,
            error_offset=DEFAULT_CURSOR_COMMIT_ERROR_GRACE_SEC - 1.0,
        )
        assert _classify(snap) == "cursor_commit_failing"

    def test_stale_error_does_not_keep_it_latched(self):
        """`record_success()` does not clear `last_error_at`.

        So an error from an hour ago must not pin the reducer to
        `cursor_commit_failing` forever once it has recovered.
        """
        snap = _snap(
            success_offset=0.0,
            advance_offset=PREVIOUS_ADVANCE_SEC,
            error_offset=DEFAULT_CURSOR_COMMIT_ERROR_GRACE_SEC + 60.0,
        )
        assert snap.last_success_at > snap.last_cursor_advance_at, "precondition: inverted"
        assert _classify(snap) == "healthy"

    def test_grace_window_is_configurable(self):
        snap = _snap(success_offset=0.0, advance_offset=30.0, error_offset=25.0)
        assert _classify(snap, cursor_commit_error_grace_sec=10.0) == "healthy"
        assert _classify(snap, cursor_commit_error_grace_sec=30.0) == "cursor_commit_failing"


class TestHigherPriorityClassificationsUnaffected:
    """Ordering inside `classify()` must not shift.

    A dead or blocked reducer outranks a commit failure; this patch touches only
    the commit-failing branch and must not reorder anything above it.
    """

    def test_dead_heartbeat_still_wins(self):
        snap = _snap(success_offset=0.0, advance_offset=PREVIOUS_ADVANCE_SEC, error_offset=1.0)
        snap.last_tick_at = datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_STALE_SEC + 10)
        assert _classify(snap) == "dead_no_heartbeat"

    def test_blocked_on_event_still_wins(self):
        snap = _snap(success_offset=0.0, advance_offset=30.0, error_offset=1.0)
        snap.blocked_event_id = "gev_stuck"
        snap.blocked_failures = 3
        assert _classify(snap) == "blocked_on_event"

    def test_disabled_still_wins(self):
        snap = _snap(success_offset=0.0, advance_offset=30.0, error_offset=1.0)
        snap.enabled = False
        assert _classify(snap) == "reducer_disabled"


def test_to_dict_threads_the_grace_through():
    """The public surface must honour the parameter, not just `classify()`."""
    snap = _snap(success_offset=0.0, advance_offset=30.0, error_offset=25.0)
    healthy = snap.to_dict(
        heartbeat_stale_sec=HEARTBEAT_STALE_SEC,
        stream_lag_degraded_sec=STREAM_LAG_DEGRADED_SEC,
        cursor_commit_error_grace_sec=10.0,
    )
    failing = snap.to_dict(
        heartbeat_stale_sec=HEARTBEAT_STALE_SEC,
        stream_lag_degraded_sec=STREAM_LAG_DEGRADED_SEC,
        cursor_commit_error_grace_sec=30.0,
    )
    assert healthy["classification"] == "healthy"
    assert failing["classification"] == "cursor_commit_failing"
