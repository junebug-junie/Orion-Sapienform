"""A budget denominated in something that can actually run out.

The old ceiling was `ewma + 3*sd` of Orion's own past demand, in units of
`risk_score` -- five hand-typed YAML constants, 67% of them 0.05. Self-sized
and fake-denominated: two independent reasons it could never bind. Live drift
across this arc: 17 -> 29 -> 347 -> 554 -> 475 -> 3,475 -> 1,787.

These tests pin the properties that make the replacement a constraint rather
than a mirror.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.autonomy.budget import (
    MIN_MEANINGFUL_ALLOWANCE_SEC,
    budget_state,
    day_elapsed_fraction,
)

DAY = 24 * 60 * 60


def _state(spent, allowance=129600.0, elapsed=0.5, enforcing=False):
    return budget_state(
        allowance_sec=allowance,
        spent_sec=spent,
        elapsed_fraction=elapsed,
        enforcing=enforcing,
    )


class TestUnconfiguredIsNotExhausted:
    """The distinction the whole thing rests on."""

    def test_no_allowance_returns_none_not_a_zero_budget(self):
        assert _state(0.0, allowance=0.0) is None
        assert _state(0.0, allowance=MIN_MEANINGFUL_ALLOWANCE_SEC - 0.01) is None

    def test_a_real_allowance_with_nothing_left_is_not_none(self):
        s = _state(200_000.0)
        assert s is not None and s.exhausted and s.remaining_sec == 0.0

    def test_negative_spend_is_refused_rather_than_clamped(self):
        with pytest.raises(ValueError):
            _state(-1.0)


class TestPaceIsTheNumberWorthWatching:
    def test_on_pace_is_one(self):
        # Half the day gone, half the allowance spent.
        s = _state(64_800.0, elapsed=0.5)
        assert s.pace == pytest.approx(1.0)
        assert s.projected_day_sec == pytest.approx(129_600.0)

    def test_burning_twice_as_fast(self):
        s = _state(64_800.0, elapsed=0.25)
        assert s.pace == pytest.approx(2.0)
        # ...and the projection says so hours before exhaustion does.
        assert not s.exhausted
        assert s.projected_day_sec == pytest.approx(259_200.0)

    def test_the_live_measurement_projects_what_was_measured(self):
        """~40 motor-hours/day was the observed draw on 2026-08-21
        (p50 5.0s/action, 1.7x concurrency). Ten hours in, that is 61,200s."""
        s = _state(61_200.0, elapsed=10 / 24)
        assert s.projected_day_sec / 3600 == pytest.approx(40.8, abs=0.1)
        assert s.pace == pytest.approx(1.13, abs=0.01)

    def test_pace_is_zero_before_the_day_starts_not_infinite(self):
        assert _state(0.0, elapsed=0.0).pace == 0.0
        assert _state(0.0, elapsed=0.0).projected_day_sec == 0.0


class TestItCanActuallyRefuse:
    def test_would_refuse_when_the_action_would_overrun(self):
        s = _state(129_599.0)
        assert s.would_refuse(5.0)
        assert not s.would_refuse(0.5)

    def test_would_refuse_is_reported_in_advisory_mode_too(self):
        """An advisory budget whose only output is 'still fine' is the
        switch-that-changes-nothing CLAUDE.md 0A bans. It has to say what it
        WOULD have stopped or there is nothing to decide the flip on."""
        s = _state(129_599.0, enforcing=False)
        assert s.mode == "advisory"
        assert s.would_refuse(5.0) is True

    def test_the_allowance_does_not_move_with_demand(self):
        """The property the old cap did not have. Spending more must not
        raise the ceiling."""
        low = _state(1_000.0)
        high = _state(120_000.0)
        assert low.allowance_sec == high.allowance_sec

    def test_a_free_action_never_refuses_against_a_live_budget(self):
        assert not _state(1_000.0).would_refuse(0.0)


class TestDayElapsed:
    def test_bounds(self):
        start = datetime(2026, 8, 22, tzinfo=timezone.utc)
        assert day_elapsed_fraction(start, start) == 0.0
        assert day_elapsed_fraction(start + timedelta(hours=12), start) == pytest.approx(0.5)
        assert day_elapsed_fraction(start + timedelta(days=3), start) == 1.0
        # A clock that went backwards clamps rather than producing a negative
        # pace denominator.
        assert day_elapsed_fraction(start - timedelta(hours=1), start) == 0.0
