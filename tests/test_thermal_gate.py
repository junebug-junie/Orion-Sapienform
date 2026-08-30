"""Hand-computed cases for the ambient thermal gate.

Every expected value below is worked out from the default thresholds by hand
(hot 32.0 / re-arm 30.5, elevated 29.5 / re-arm 28.0), not read back off the
implementation.
"""

from __future__ import annotations

import pytest

from orion.autonomy.thermal_gate import (
    DEFAULT_HOT_C,
    DEFAULT_MAX_READING_AGE_SEC,
    thermal_state,
)


class TestPlainClassification:
    @pytest.mark.parametrize(
        "temp_c,expected",
        [
            (21.0, "normal"),
            (29.4, "normal"),
            (29.5, "elevated"),  # trip is >=
            (30.74, "elevated"),  # the live reading on 2026-08-30
            (31.9, "elevated"),
            (32.0, "hot"),  # trip is >=
            (34.0, "hot"),  # what Juniper reported in the office
        ],
    )
    def test_state_from_a_cold_start(self, temp_c: float, expected: str) -> None:
        assert thermal_state(temp_c=temp_c, age_sec=1.0).state == expected

    def test_only_hot_blocks_gpu_work(self) -> None:
        assert thermal_state(temp_c=30.74, age_sec=1.0).allows_gpu_work is True
        assert thermal_state(temp_c=34.0, age_sec=1.0).allows_gpu_work is False


class TestHysteresis:
    """A bare `temp > threshold` on a wandering reading flaps every tick, and a
    gate that flaps is worse than none: the work still happens, unpredictably."""

    def test_stays_hot_between_rearm_and_trip(self) -> None:
        # 31.0 is below the 32.0 trip but above the 30.5 re-arm.
        assert thermal_state(temp_c=31.0, age_sec=1.0, previous_state="hot").state == "hot"
        # Same reading from a cold start is merely elevated -- which is exactly
        # the difference hysteresis buys, and a stateless gate cannot express.
        assert thermal_state(temp_c=31.0, age_sec=1.0, previous_state="normal").state == "elevated"

    def test_rearms_only_once_genuinely_cooler(self) -> None:
        assert thermal_state(temp_c=30.5, age_sec=1.0, previous_state="hot").state == "elevated"
        assert thermal_state(temp_c=30.4, age_sec=1.0, previous_state="hot").state == "elevated"

    def test_elevated_also_rearms_cooler_than_it_trips(self) -> None:
        assert thermal_state(temp_c=28.5, age_sec=1.0, previous_state="elevated").state == "elevated"
        assert thermal_state(temp_c=28.5, age_sec=1.0, previous_state="normal").state == "normal"
        assert thermal_state(temp_c=28.0, age_sec=1.0, previous_state="elevated").state == "normal"

    def test_a_reading_oscillating_across_the_trip_does_not_flap(self) -> None:
        """The real failure mode, walked step by step."""
        readings = [31.8, 32.1, 31.7, 32.0, 31.6, 30.9, 30.6]
        state = "normal"
        states = []
        for temp in readings:
            state = thermal_state(temp_c=temp, age_sec=1.0, previous_state=state).state
            states.append(state)
        # Trips at 32.1 and then HOLDS -- no return to elevated while the room
        # sits between the re-arm point and the trip.
        assert states == ["elevated", "hot", "hot", "hot", "hot", "hot", "hot"]


class TestDegradedReadings:
    """`unknown` must never be reported as a clean allow -- an absent reading is
    not evidence the room is cool."""

    def test_missing_reading_is_unknown_and_allows(self) -> None:
        verdict = thermal_state(temp_c=None, age_sec=None)
        assert verdict.state == "unknown"
        assert verdict.degraded is True
        assert verdict.allows_gpu_work is True
        assert verdict.reason == "no_reading"

    def test_stale_reading_is_unknown_even_when_it_reads_hot(self) -> None:
        """A hot reading from an hour ago says nothing about the room now, and
        must not latch the gate closed on a dead sensor."""
        verdict = thermal_state(temp_c=40.0, age_sec=DEFAULT_MAX_READING_AGE_SEC + 1)
        assert verdict.state == "unknown"
        assert verdict.degraded is True
        assert verdict.allows_gpu_work is True
        assert "stale" in verdict.reason

    def test_a_fresh_reading_at_the_age_limit_is_still_used(self) -> None:
        verdict = thermal_state(temp_c=34.0, age_sec=DEFAULT_MAX_READING_AGE_SEC)
        assert verdict.state == "hot"
        assert verdict.degraded is False

    def test_a_normal_verdict_is_not_degraded(self) -> None:
        assert thermal_state(temp_c=21.0, age_sec=1.0).degraded is False


def test_the_hot_threshold_sits_below_the_reported_room_temperature() -> None:
    """A threshold nothing ever crosses is a switch that changes nothing. The
    office was reported at ~34C, so the trip has to be under that to ever fire."""
    assert DEFAULT_HOT_C < 34.0


def test_reason_carries_the_number_that_caused_the_verdict() -> None:
    """A refusal that does not say what it read cannot be argued with or
    debugged."""
    verdict = thermal_state(temp_c=33.2, age_sec=1.0)
    assert "33.2" in verdict.reason
    assert "32.0" in verdict.reason
