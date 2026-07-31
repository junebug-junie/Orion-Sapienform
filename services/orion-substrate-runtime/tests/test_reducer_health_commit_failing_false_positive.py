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
    CURSOR_COMMIT_ERROR_GRACE_SEC,
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
    def test_in_flight_batch_with_no_error_is_healthy(self):
        """The state that paged: mid-batch, cursor not yet advanced, zero errors.

        Review correction 2026-07-31: an earlier version of this test quoted
        `last_success_at 20:28:51.632147 / last_cursor_advance_at 20:28:52.043960`
        as "the exact live window". Those are 411ms apart with ADVANCE LATER, so
        they are not inverted at all -- that is the recovered post-advance state,
        healthy under the old predicate too. The state that actually paged is a
        sample taken BEFORE the advance landed, where the newest advance on record
        still belongs to the previous batch. That is what this builds.
        """
        snap = _snap(
            success_offset=0.0,
            advance_offset=PREVIOUS_ADVANCE_SEC,
            error_offset=None,
        )
        assert snap.last_success_at > snap.last_cursor_advance_at, "precondition: inverted"
        assert _classify(snap) == "healthy"

    @pytest.mark.parametrize("batch_gap_sec", [4.0, 13.0, 28.1, 300.0])
    def test_any_inversion_width_without_an_error_is_healthy(self, batch_gap_sec):
        """How long the batch has been in flight is irrelevant when nothing failed.

        Review correction 2026-07-31: this previously varied a `window_sec` that
        was then ADDED to `PREVIOUS_ADVANCE_SEC`, so all four cases produced a
        ~20s inversion and tested the same thing. The predicate never inspects
        the gap width at all, so the meaningful axis is how stale the previous
        advance is. Values are the live-measured spread of real batch gaps
        (4-13s typical, 28.1s the observed max on route), plus 300s as a
        deliberately extreme case.
        """
        snap = _snap(
            success_offset=0.0,
            advance_offset=batch_gap_sec,
            error_offset=None,
        )
        assert snap.last_success_at > snap.last_cursor_advance_at, "precondition: inverted"
        assert _classify(snap) == "healthy"


class TestRealCommitFailureStillDetected:
    def test_tick_failure_between_success_and_advance_is_still_a_failure(self):
        """The condition that genuinely reaches this branch.

        Review correction 2026-07-31: this test previously justified itself with
        `_advance_cursor()`'s own failure paths. Those pass a real `event_id` to
        `record_error()`, which sets `blocked_event_id`, so `blocked_on_event`
        returns first and they NEVER reach this branch (see
        `test_advance_cursor_failure_is_blocked_on_event_not_commit_failing`).

        What does reach it is the poll loops' own `record_error(event_id=None)`
        after a tick raised BETWEEN `record_success()` and the advance -- a failed
        `publish_accepted_events`, `save_execution_trajectory`, or
        `_write_prediction_error_node`. `blocked_event_id` stays None there, which
        is exactly the state built here.
        """
        snap = _snap(success_offset=0.0, advance_offset=30.0, error_offset=0.5)
        assert snap.blocked_event_id is None, "precondition: the event_id=None path"
        assert _classify(snap) == "cursor_commit_failing"

    def test_advance_cursor_failure_is_blocked_on_event_not_commit_failing(self):
        """Pin the precedence rule, so nobody 'simplifies' the ordering.

        Replaying the real sequence from `worker.py::_advance_cursor` -- which
        passes a real event_id -- against the live module gives `blocked_on_event`,
        both before and after this patch. A genuinely stuck cursor commit pages as
        `reducer_blocked:<cursor>`, never `reducer_cursor_commit_failing:<cursor>`,
        despite the latter's name. Detection is not lost; it just lives elsewhere.
        """
        snap = _snap(success_offset=0.0, advance_offset=30.0, error_offset=0.5)
        snap.blocked_event_id = "gev_missing"
        snap.blocked_failures = 1
        assert _classify(snap) == "blocked_on_event"

    def test_error_just_inside_the_grace_window_still_fires(self):
        snap = _snap(
            success_offset=0.0,
            advance_offset=300.0,
            error_offset=CURSOR_COMMIT_ERROR_GRACE_SEC - 1.0,
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
            error_offset=CURSOR_COMMIT_ERROR_GRACE_SEC + 60.0,
        )
        assert snap.last_success_at > snap.last_cursor_advance_at, "precondition: inverted"
        assert _classify(snap) == "healthy"

    def test_grace_boundary_is_inclusive(self):
        """The predicate is `<= grace`; pin that rather than only testing near it."""
        just_inside = _snap(
            success_offset=0.0,
            advance_offset=300.0,
            error_offset=CURSOR_COMMIT_ERROR_GRACE_SEC - 0.5,
        )
        just_outside = _snap(
            success_offset=0.0,
            advance_offset=300.0,
            error_offset=CURSOR_COMMIT_ERROR_GRACE_SEC + 0.5,
        )
        assert _classify(just_inside) == "cursor_commit_failing"
        assert _classify(just_outside) == "healthy"


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


def test_benign_inversion_with_stream_lag_now_reports_alive_behind():
    """A behavior change worth pinning: it used to mask a real lag signal.

    Inverted + no error + stream lag over the threshold previously returned
    `cursor_commit_failing`, which hid the fact that the reducer was genuinely
    behind. It now falls through to the lag branch and says so.
    """
    snap = _snap(success_offset=0.0, advance_offset=PREVIOUS_ADVANCE_SEC, error_offset=None)
    snap.stream_lag_sec = STREAM_LAG_DEGRADED_SEC + 10.0
    assert snap.last_success_at > snap.last_cursor_advance_at, "precondition: inverted"
    assert _classify(snap) == "alive_behind"


def test_to_dict_agrees_with_classify():
    """`to_dict()` is the surface /grammar/truth actually reads."""
    snap = _snap(success_offset=0.0, advance_offset=PREVIOUS_ADVANCE_SEC, error_offset=None)
    out = snap.to_dict(
        heartbeat_stale_sec=HEARTBEAT_STALE_SEC,
        stream_lag_degraded_sec=STREAM_LAG_DEGRADED_SEC,
    )
    assert out["classification"] == "healthy" == _classify(snap)
