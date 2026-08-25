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
from types import SimpleNamespace

import pytest

from app.worker import THEATER_TRIPWIRE_UNPRODUCTIVE_THRESHOLD, THEATER_TRIPWIRE_WINDOW, ExecutionDispatchRuntimeWorker


def _bare_worker() -> ExecutionDispatchRuntimeWorker:
    worker = object.__new__(ExecutionDispatchRuntimeWorker)
    worker.theater_tripwire_active = False
    worker._recent_dispatch_statuses = deque(maxlen=THEATER_TRIPWIRE_WINDOW)
    worker._notify = None  # _notify_tripwire wraps its use in try/except
    # 2026-08-25: tripping now also arms the recovery schedule, so the trip
    # path reads settings and the clock. Recovery BEHAVIOUR is exercised in
    # test_tripwire_recovery.py; this file still covers only the trip
    # predicate, and these are the minimum attributes that path touches.
    worker._settings = SimpleNamespace(
        orion_dispatch_tripwire_probe_enabled=True,
        orion_dispatch_tripwire_probe_cooldown_sec=300.0,
        orion_dispatch_tripwire_probe_max_cooldown_sec=3600.0,
        orion_dispatch_tripwire_rearm_successes=3,
    )
    worker._tick_dispatch_statuses = []
    worker._tripwire_tripped_at = None
    worker._tripwire_next_probe_at = None
    worker._tripwire_probe_cooldown_sec = 300.0
    worker._tripwire_probe_successes = 0
    worker._tripwire_probe_attempts = 0
    worker._tripwire_last_blocked_log_at = None
    worker._tripwire_last_notify_at = None
    worker._tripwire_probe_in_flight = False
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
        n_empty = int(THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_UNPRODUCTIVE_THRESHOLD) + 1
        for _ in range(n_empty):
            worker._recent_dispatch_statuses.append("empty")
        for _ in range(THEATER_TRIPWIRE_WINDOW - n_empty):
            worker._recent_dispatch_statuses.append("success")
        assert worker._check_theater_tripwire() is True

    def test_does_not_trip_at_exactly_the_threshold_fraction(self) -> None:
        """The real condition is `>`, not `>=` -- exactly half empty (the
        literal threshold value) must not trip."""
        worker = _bare_worker()
        half = int(THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_UNPRODUCTIVE_THRESHOLD)
        for _ in range(half):
            worker._recent_dispatch_statuses.append("empty")
        for _ in range(THEATER_TRIPWIRE_WINDOW - half):
            worker._recent_dispatch_statuses.append("success")
        assert worker._check_theater_tripwire() is False

    def test_does_not_self_clear_once_tripped(self) -> None:
        """This PREDICATE only ever sets the latch; it never clears it.

        Narrowed 2026-08-25. It used to read "once tripped, stays tripped for
        this process's lifetime -- only a real restart re-arms it", and that
        rule cost 45 hours of zero dispatches on 2026-08-23. The latch can now
        be cleared, but only by a run of consecutive successful recovery probes
        in `_evaluate_tripwire_probe` (see test_tripwire_recovery.py).

        What this test still pins is the part that must not change: a healthy
        trailing WINDOW is not by itself grounds to resume. That is the
        original design's "could silently resume sending on a coincidentally-
        good sample" objection, and it is still answered here."""
        worker = _bare_worker()
        n_empty = int(THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_UNPRODUCTIVE_THRESHOLD) + 1
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

    def test_failed_status_now_counts_as_unproductive(self) -> None:
        """BEHAVIOUR REVERSED 2026-08-13, deliberately.

        This test previously asserted that "failed" must NOT contribute to the
        count, on the reasoning that an RPC/send error is a different failure
        mode than a hollow-but-successful response.

        That reasoning stopped holding when skill-verb results started being
        classified honestly. Before that change, a `skills.runtime.*` verb that
        failed was stored as "empty" and therefore counted here. After it, the
        same failure is stored as "failed" -- so under the old predicate a
        maintenance verb could fail on every one of the trailing 10 dispatches
        and never trip the one mechanism built to catch a dead motor nerve.

        The predicate now counts every non-success status, which restores that
        coverage and additionally covers RPC send failures. Live trip risk was
        measured before shipping rather than assumed: 30 failed against 12,782
        success over 24h (0.23%), versus a threshold of >50% of trailing 10.
        """
        worker = _bare_worker()
        for _ in range(THEATER_TRIPWIRE_WINDOW):
            worker._recent_dispatch_statuses.append("failed")
        assert worker._check_theater_tripwire() is True

    def test_all_success_never_trips(self) -> None:
        """The inverse guard. Without it, a predicate that counted everything
        (`sum(1 for _ in recent)`) would pass every other test in this class."""
        worker = _bare_worker()
        for _ in range(THEATER_TRIPWIRE_WINDOW):
            worker._recent_dispatch_statuses.append("success")
        assert worker._check_theater_tripwire() is False

    def test_mixed_empty_and_failed_combine_toward_the_threshold(self) -> None:
        """Hand-derived: window 10, threshold 0.5, so >5 unproductive trips.

        3 empty + 3 failed = 6 unproductive > 5, with 4 success. Neither
        status alone reaches the threshold; only counting them together does.
        This is precisely the mixed-failure case the old predicate could not
        see at all.
        """
        worker = _bare_worker()
        for _ in range(3):
            worker._recent_dispatch_statuses.append("empty")
        for _ in range(3):
            worker._recent_dispatch_statuses.append("failed")
        for _ in range(4):
            worker._recent_dispatch_statuses.append("success")
        assert len(worker._recent_dispatch_statuses) == THEATER_TRIPWIRE_WINDOW
        assert worker._check_theater_tripwire() is True
