"""Unit tests for the execution-dispatch-runtime theater tripwire's in-process
window (worker.py's `_check_theater_tripwire`).

Constructs `ExecutionDispatchRuntimeWorker` via `object.__new__` rather than
`__init__` -- the real constructor builds a live Postgres store, loads a policy
YAML off disk, and constructs a NotifyClient, none of which this pure logic
test needs or should depend on. Only the attributes `_check_theater_tripwire`
and `_notify_tripwire` actually touch are set.
"""

from __future__ import annotations

from collections import deque

import pytest

from app.worker import THEATER_TRIPWIRE_EMPTY_THRESHOLD, THEATER_TRIPWIRE_WINDOW, ExecutionDispatchRuntimeWorker


def _bare_worker() -> ExecutionDispatchRuntimeWorker:
    worker = object.__new__(ExecutionDispatchRuntimeWorker)
    worker.theater_tripwire_active = False
    worker._recent_dispatch_statuses = deque(maxlen=THEATER_TRIPWIRE_WINDOW)
    worker._notify = None  # _notify_tripwire wraps its use in try/except
    return worker


class TestTheaterTripwireInProcessWindow:
    def test_fresh_process_is_not_tripped_regardless_of_postgres_history(self) -> None:
        """The regression case for the real 2026-07-25 incident: a fresh
        process (empty in-memory deque) must read as not-tripped even if
        substrate_dispatch_results (Postgres, not touched by this test at
        all) still holds stale rows from before this restart -- the whole
        point of moving off a live Postgres query.
        """
        worker = _bare_worker()
        assert worker._check_theater_tripwire() is False

    def test_fewer_than_window_statuses_does_not_trip(self) -> None:
        worker = _bare_worker()
        for _ in range(THEATER_TRIPWIRE_WINDOW - 1):
            worker._recent_dispatch_statuses.append("empty")
        assert worker._check_theater_tripwire() is False

    def test_trips_when_more_than_threshold_fraction_is_empty(self) -> None:
        worker = _bare_worker()
        n_empty = int(THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_EMPTY_THRESHOLD) + 1
        for _ in range(n_empty):
            worker._recent_dispatch_statuses.append("empty")
        for _ in range(THEATER_TRIPWIRE_WINDOW - n_empty):
            worker._recent_dispatch_statuses.append("success")
        assert worker._check_theater_tripwire() is True

    def test_does_not_trip_at_exactly_the_threshold_fraction(self) -> None:
        """The real condition is `>`, not `>=` -- exactly half empty (the
        literal threshold value) must not trip."""
        worker = _bare_worker()
        half = int(THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_EMPTY_THRESHOLD)
        for _ in range(half):
            worker._recent_dispatch_statuses.append("empty")
        for _ in range(THEATER_TRIPWIRE_WINDOW - half):
            worker._recent_dispatch_statuses.append("success")
        assert worker._check_theater_tripwire() is False

    def test_does_not_self_clear_once_tripped(self) -> None:
        """Deliberate design (worker.py's own docstring): once tripped, stays
        tripped for this process's lifetime even if fresh appends show a
        healthy window -- only a real restart (a fresh process, fresh deque)
        re-arms it. This is NOT a regression to fix; it's the property this
        patch is careful to preserve while fixing the separate, real bug
        (stale pre-restart Postgres rows defeating restart-to-re-arm)."""
        worker = _bare_worker()
        n_empty = int(THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_EMPTY_THRESHOLD) + 1
        for _ in range(n_empty):
            worker._recent_dispatch_statuses.append("empty")
        for _ in range(THEATER_TRIPWIRE_WINDOW - n_empty):
            worker._recent_dispatch_statuses.append("success")
        assert worker._check_theater_tripwire() is True

        # Window now fills with nothing but healthy results -- old entries
        # fall off the maxlen deque, but the process-level latch must still
        # hold.
        for _ in range(THEATER_TRIPWIRE_WINDOW):
            worker._recent_dispatch_statuses.append("success")
        assert worker._check_theater_tripwire() is True

    def test_deque_is_bounded_to_window_size(self) -> None:
        """maxlen enforces the window itself -- appending well beyond
        THEATER_TRIPWIRE_WINDOW must never grow the deque past it."""
        worker = _bare_worker()
        for _ in range(THEATER_TRIPWIRE_WINDOW * 3):
            worker._recent_dispatch_statuses.append("success")
        assert len(worker._recent_dispatch_statuses) == THEATER_TRIPWIRE_WINDOW

    def test_failed_status_does_not_count_as_empty(self) -> None:
        """`_check_theater_tripwire` only counts literal "empty" -- "failed"
        (an RPC/send error, a different failure mode than a hollow-but-
        technically-successful response) must not contribute to the empty
        count, matching pre-existing behavior this patch does not change."""
        worker = _bare_worker()
        for _ in range(THEATER_TRIPWIRE_WINDOW):
            worker._recent_dispatch_statuses.append("failed")
        assert worker._check_theater_tripwire() is False
