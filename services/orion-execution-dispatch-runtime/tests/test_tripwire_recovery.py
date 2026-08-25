"""The theater tripwire must be able to let go.

Regression tests for the 2026-08-23 outage: 45 hours of zero dispatches after
ordinary post-redeploy startup wobble latched a tripwire that had no code path
back to False. Three defects, tested separately below:

  1. no re-arm path at all
  2. self-sealing -- once tripped, no new evidence could ever be gathered
  3. silent -- one warning at the trip, then nothing for 45 hours

Built with `object.__new__` for the same reason as test_theater_tripwire.py:
the real constructor wants a live Postgres store, a policy YAML and a
NotifyClient, none of which this logic needs.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from app.worker import (
    THEATER_TRIPWIRE_WINDOW,
    TRIPWIRE_BLOCKED_LOG_INTERVAL_SEC,
    TRIPWIRE_BLOCKED_WARNING,
    TRIPWIRE_PROBE_BACKOFF_FACTOR,
    TRIPWIRE_PROBE_DISPATCHES,
    TRIPWIRE_RENOTIFY_INTERVAL_SEC,
    ExecutionDispatchRuntimeWorker,
)

COOLDOWN = 300.0
MAX_COOLDOWN = 3600.0
REARM = 3


class _Clock:
    """Controllable stand-in for time.monotonic."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _worker(
    *,
    probe_enabled: bool = True,
    rearm: int = REARM,
    cooldown: float = COOLDOWN,
) -> ExecutionDispatchRuntimeWorker:
    w = object.__new__(ExecutionDispatchRuntimeWorker)
    w._settings = SimpleNamespace(
        orion_dispatch_tripwire_probe_enabled=probe_enabled,
        orion_dispatch_tripwire_probe_cooldown_sec=cooldown,
        orion_dispatch_tripwire_probe_max_cooldown_sec=MAX_COOLDOWN,
        orion_dispatch_tripwire_rearm_successes=rearm,
    )
    w.theater_tripwire_active = False
    w._recent_dispatch_statuses = deque(maxlen=THEATER_TRIPWIRE_WINDOW)
    w._tick_dispatch_statuses = []
    w._tripwire_tripped_at = None
    w._tripwire_next_probe_at = None
    w._tripwire_probe_cooldown_sec = cooldown
    w._tripwire_probe_successes = 0
    w._tripwire_probe_attempts = 0
    w._tripwire_last_blocked_log_at = None
    w._tripwire_last_notify_at = None
    w._tripwire_probe_in_flight = False
    w._notify = None  # every _notify_* wraps its use in try/except

    clock = _Clock()
    w._monotonic = clock  # type: ignore[method-assign]
    w._clock = clock  # test handle
    return w


def _trip(w: ExecutionDispatchRuntimeWorker) -> None:
    """Trip via the real predicate, not by poking the flag."""
    for _ in range(THEATER_TRIPWIRE_WINDOW):
        w._record_dispatch_status("failed")
    assert w._check_theater_tripwire() is True


def _probe(w: ExecutionDispatchRuntimeWorker, *statuses: str) -> None:
    """Run one probe tick that records `statuses`, then evaluate it."""
    slots = w._claim_tripwire_probe_slots()
    assert slots > 0, "expected the cooldown to have elapsed"
    w._tripwire_probe_in_flight = True
    w._tick_dispatch_statuses = []
    for s in statuses:
        w._record_dispatch_status(s)
    w._evaluate_tripwire_probe()


class TestDefect1NoRearmPath:
    def test_a_run_of_successful_probes_clears_the_latch(self) -> None:
        w = _worker()
        _trip(w)
        for i in range(REARM):
            assert w.theater_tripwire_active is True, f"cleared early at probe {i}"
            w._clock.advance(COOLDOWN)
            _probe(w, "success")
        assert w.theater_tripwire_active is False

    def test_one_success_short_of_the_run_does_not_clear(self) -> None:
        w = _worker()
        _trip(w)
        for _ in range(REARM - 1):
            w._clock.advance(COOLDOWN)
            _probe(w, "success")
        assert w.theater_tripwire_active is True
        assert w._tripwire_probe_successes == REARM - 1

    def test_a_failure_resets_the_run_to_zero(self) -> None:
        """The original design's stated objection, answered: a coincidentally
        good sample cannot accumulate toward re-arming across a flapping motor.
        """
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN)
        _probe(w, "success")
        w._clock.advance(COOLDOWN)
        _probe(w, "success")
        assert w._tripwire_probe_successes == 2

        w._clock.advance(w._tripwire_probe_cooldown_sec)
        _probe(w, "failed")
        assert w._tripwire_probe_successes == 0
        assert w.theater_tripwire_active is True

    def test_alternating_success_failure_never_re_arms(self) -> None:
        w = _worker()
        _trip(w)
        for _ in range(20):
            w._clock.advance(MAX_COOLDOWN)
            _probe(w, "success")
            w._clock.advance(MAX_COOLDOWN)
            _probe(w, "failed")
        assert w.theater_tripwire_active is True

    def test_probing_can_be_disabled_restoring_restart_only_recovery(self) -> None:
        w = _worker(probe_enabled=False)
        _trip(w)
        w._clock.advance(MAX_COOLDOWN * 10)
        assert w._claim_tripwire_probe_slots() == 0
        assert w.theater_tripwire_active is True

    def test_rearm_of_one_is_permitted_but_is_not_the_default(self) -> None:
        w = _worker(rearm=1)
        _trip(w)
        w._clock.advance(COOLDOWN)
        _probe(w, "success")
        assert w.theater_tripwire_active is False
        assert REARM > 1, "the shipped default must not be a single-sample re-arm"


class TestDefect2SelfSealing:
    def test_the_frozen_pre_trip_window_cannot_vote_on_recovery(self) -> None:
        """At probe time `_recent_dispatch_statuses` is still 10/10 failures.
        Judging a probe on that window would make recovery impossible; the probe
        must be judged only on what the probe itself produced.
        """
        w = _worker()
        _trip(w)
        assert all(s == "failed" for s in w._recent_dispatch_statuses)
        for _ in range(REARM):
            w._clock.advance(COOLDOWN)
            _probe(w, "success")
        assert w.theater_tripwire_active is False

    def test_clearing_does_not_wipe_the_trailing_window(self) -> None:
        """A motor that passes its probes but is still mostly failing must stay
        visible to the ordinary tripwire predicate.
        """
        w = _worker()
        _trip(w)
        for _ in range(REARM):
            w._clock.advance(COOLDOWN)
            _probe(w, "success")
        assert w.theater_tripwire_active is False
        # 10-slot deque: 7 pre-trip failures survive alongside the 3 probes.
        assert list(w._recent_dispatch_statuses).count("failed") == 7

    def test_re_tripping_after_recovery_works_normally(self) -> None:
        w = _worker()
        _trip(w)
        for _ in range(REARM):
            w._clock.advance(COOLDOWN)
            _probe(w, "success")
        assert w.theater_tripwire_active is False
        for _ in range(THEATER_TRIPWIRE_WINDOW):
            w._record_dispatch_status("failed")
        assert w._check_theater_tripwire() is True
        assert w._tripwire_probe_attempts == 0, "recovery state must reset on a re-trip"


class TestCooldownAndBackoff:
    def test_no_probe_before_the_first_cooldown_elapses(self) -> None:
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN - 1.0)
        assert w._claim_tripwire_probe_slots() == 0
        w._clock.advance(1.0)
        assert w._claim_tripwire_probe_slots() == TRIPWIRE_PROBE_DISPATCHES

    def test_failed_probes_back_off_exponentially(self) -> None:
        w = _worker()
        _trip(w)
        expected = COOLDOWN
        for _ in range(4):
            w._clock.advance(w._tripwire_probe_cooldown_sec)
            _probe(w, "failed")
            expected = min(expected * TRIPWIRE_PROBE_BACKOFF_FACTOR, MAX_COOLDOWN)
            assert w._tripwire_probe_cooldown_sec == pytest.approx(expected)

    def test_backoff_is_capped(self) -> None:
        w = _worker()
        _trip(w)
        for _ in range(30):
            w._clock.advance(w._tripwire_probe_cooldown_sec)
            _probe(w, "failed")
        assert w._tripwire_probe_cooldown_sec == MAX_COOLDOWN

    def test_a_dead_motor_costs_at_most_one_action_per_max_cooldown(self) -> None:
        """The bounded-cost claim, asserted rather than described: over a
        simulated day against a permanently broken motor, the probe machinery
        must not be able to spend more than 24h/max_cooldown actions.
        """
        w = _worker()
        _trip(w)
        elapsed = 0.0
        sent = 0
        while elapsed < 86_400.0:
            w._clock.advance(60.0)
            elapsed += 60.0
            slots = w._claim_tripwire_probe_slots()
            if slots > 0:
                sent += slots
                w._tripwire_probe_in_flight = True
                w._tick_dispatch_statuses = []
                w._record_dispatch_status("failed")
                w._evaluate_tripwire_probe()
        # Ceiling: the first few attempts happen while the backoff is still
        # widening, then it settles at one per hour.
        assert sent <= 24 + 4, f"probe budget ran away: {sent} actions in 24h"
        assert sent >= 20, f"backoff starved recovery entirely: only {sent} probes"

    def test_cooldown_resets_after_a_successful_recovery(self) -> None:
        w = _worker()
        _trip(w)
        for _ in range(3):
            w._clock.advance(w._tripwire_probe_cooldown_sec)
            _probe(w, "failed")
        assert w._tripwire_probe_cooldown_sec > COOLDOWN
        for _ in range(REARM):
            w._clock.advance(w._tripwire_probe_cooldown_sec)
            _probe(w, "success")
        assert w.theater_tripwire_active is False
        assert w._tripwire_probe_cooldown_sec == COOLDOWN


class TestProbeEvidenceRules:
    def test_a_probe_that_recorded_nothing_is_not_a_success(self) -> None:
        """'Nothing came back' is the exact condition the tripwire exists for.
        Counting it toward re-arming would let a fully dead motor talk its way
        back into service.
        """
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN)
        slots = w._claim_tripwire_probe_slots()
        assert slots > 0
        w._tripwire_probe_in_flight = True
        w._tick_dispatch_statuses = []
        w._evaluate_tripwire_probe()
        assert w._tripwire_probe_successes == 0
        assert w.theater_tripwire_active is True

    def test_empty_status_counts_as_a_failed_probe(self) -> None:
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN)
        _probe(w, "empty")
        assert w._tripwire_probe_successes == 0

    def test_a_refunded_probe_does_not_consume_an_attempt(self) -> None:
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN)
        assert w._claim_tripwire_probe_slots() > 0
        assert w._tripwire_probe_attempts == 1
        w._tripwire_probe_in_flight = True
        w._refund_tripwire_probe()
        assert w._tripwire_probe_attempts == 0
        # And the next tick may probe immediately -- no evidence was gathered,
        # so no backoff is owed.
        assert w._claim_tripwire_probe_slots() > 0

    def test_a_partially_successful_probe_is_a_failed_probe(self) -> None:
        """Pins `all(...)`, not `any(...)`.

        Found by mutation testing: with TRIPWIRE_PROBE_DISPATCHES == 1 a probe
        only ever carries one status, so `any` and `all` are indistinguishable
        and every other test in this file passes under either. That equivalence
        is an accident of the current width -- raise the constant and `any`
        silently starts re-arming on a probe that half-failed. Exercised
        directly against the evaluator so it holds at any width.
        """
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN)
        _probe(w, "success", "failed")
        assert w._tripwire_probe_successes == 0
        assert w.theater_tripwire_active is True

    def test_a_fully_successful_multi_candidate_probe_counts(self) -> None:
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN)
        _probe(w, "success", "success")
        assert w._tripwire_probe_successes == 1

    def test_probe_sends_exactly_one_candidate(self) -> None:
        w = _worker()
        _trip(w)
        w._clock.advance(COOLDOWN)
        assert w._claim_tripwire_probe_slots() == 1

    def test_record_dispatch_status_writes_both_windows(self) -> None:
        """The single choke point. A future call site that updated only one of
        these would silently break either the probe or the tripwire itself.
        """
        w = _worker()
        w._record_dispatch_status("success")
        assert list(w._recent_dispatch_statuses) == ["success"]
        assert w._tick_dispatch_statuses == ["success"]


class TestDefect3Silence:
    def test_every_blocked_tick_carries_a_frame_warning(self) -> None:
        w = _worker()
        _trip(w)
        frame = SimpleNamespace(warnings=["pre-existing"], candidates=[1, 2, 3])
        for _ in range(5):
            warnings = w._tripwire_blocked_warnings(frame)
            assert "pre-existing" in warnings
            assert any(x.startswith(TRIPWIRE_BLOCKED_WARNING) for x in warnings)

    def test_the_warning_states_how_long_and_how_many_were_withheld(self) -> None:
        w = _worker()
        _trip(w)
        w._clock.advance(7_200.0)
        frame = SimpleNamespace(warnings=[], candidates=[1, 2, 3, 4])
        warning = w._tripwire_blocked_warnings(frame)[0]
        assert "tripped_for_sec=7200" in warning
        assert "candidates_withheld=4" in warning

    def test_the_blocked_log_is_throttled_not_silent(self) -> None:
        w = _worker()
        _trip(w)
        frame = SimpleNamespace(warnings=[], candidates=[])

        w._tripwire_blocked_warnings(frame)
        first = w._tripwire_last_blocked_log_at
        assert first is not None

        w._clock.advance(TRIPWIRE_BLOCKED_LOG_INTERVAL_SEC / 2)
        w._tripwire_blocked_warnings(frame)
        assert w._tripwire_last_blocked_log_at == first, "logged inside the throttle"

        w._clock.advance(TRIPWIRE_BLOCKED_LOG_INTERVAL_SEC)
        w._tripwire_blocked_warnings(frame)
        assert w._tripwire_last_blocked_log_at != first

    def test_renotifies_on_an_interval_rather_than_once(self) -> None:
        """45 hours ran on a single fire-once warning. Count real calls."""
        w = _worker()
        _trip(w)
        sent: list[float] = []
        w._notify_tripwire_still_blocked = lambda secs: sent.append(secs)  # type: ignore[method-assign]

        frame = SimpleNamespace(warnings=[], candidates=[])
        # Simulate 5 hours of blocked ticks at a 2s cadence.
        for _ in range(5 * 3600 // 2):
            w._tripwire_blocked_warnings(frame)
            w._clock.advance(2.0)

        assert len(sent) == pytest.approx(5, abs=1), f"expected ~hourly, got {len(sent)}"
        assert sent == sorted(sent)
        assert sent[0] >= TRIPWIRE_RENOTIFY_INTERVAL_SEC

    def test_the_trip_itself_still_notifies_immediately(self) -> None:
        w = _worker()
        calls: list[tuple[int, int]] = []
        w._notify_tripwire = lambda a, b: calls.append((a, b))  # type: ignore[method-assign]
        _trip(w)
        assert len(calls) == 1

    def test_clearing_notifies(self) -> None:
        w = _worker()
        _trip(w)
        cleared: list[tuple[float, int]] = []
        w._notify_tripwire_cleared = lambda secs, n: cleared.append((secs, n))  # type: ignore[method-assign]
        for _ in range(REARM):
            w._clock.advance(COOLDOWN)
            _probe(w, "success")
        assert len(cleared) == 1
        assert cleared[0][1] == REARM


class TestTheIncident:
    def test_the_2026_08_23_outage_would_now_self_recover(self) -> None:
        """End-to-end shape of the real incident: a transient burst of failures
        during a redeploy, a motor that is fine again minutes later, and no
        human available to restart anything.
        """
        w = _worker()

        # 07:33 -- six fast plan_status=fail land on top of four partials.
        for _ in range(4):
            w._record_dispatch_status("empty")
        for _ in range(6):
            w._record_dispatch_status("failed")
        assert w._check_theater_tripwire() is True

        # cortex-exec is healthy again within a couple of minutes, but nothing
        # notices: under the old code this state persisted for 45 hours.
        recovered_after_sec = 0.0
        frame = SimpleNamespace(warnings=[], candidates=[1, 2, 3, 4, 5])
        for _ in range(4 * 3600 // 2):  # up to 4 simulated hours, 2s ticks
            if not w.theater_tripwire_active:
                break
            slots = w._claim_tripwire_probe_slots()
            if slots > 0:
                w._tripwire_probe_in_flight = True
                w._tick_dispatch_statuses = []
                w._record_dispatch_status("success")
                w._evaluate_tripwire_probe()
            else:
                w._tripwire_blocked_warnings(frame)
            w._clock.advance(2.0)
            recovered_after_sec += 2.0

        assert w.theater_tripwire_active is False, "still latched after 4 hours"
        # Three probes, one cooldown apart: recovery inside the hour, versus 45
        # hours of nothing.
        assert recovered_after_sec <= REARM * COOLDOWN + 60.0
        assert recovered_after_sec < 3_600.0
