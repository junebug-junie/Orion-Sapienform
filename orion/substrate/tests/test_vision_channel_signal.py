"""Unit tests for the perceptual-health signals feeding capability:vision.

An availability channel (is the eye producing at all) and a yield channel (is
what it produces carrying anything). Both read the detector's own output; the
earlier bus-cadence version was deleted, not tuned -- see
orion/substrate/prediction_error.py's capability:vision section.
"""

from __future__ import annotations

import math

from orion.substrate.prediction_error import (
    perceptual_blindness_pressure,
    perceptual_yield,
    vision_channel_staleness_pressure,
)


# --- availability channel -------------------------------------------------


def test_rest_point_is_exactly_zero_and_reachable() -> None:
    """
    The whole point of gate item 4. At health the newest vision message is a
    fraction of a second old (measured live: frames every 0.1s), so this must
    read a clean 0.0 rather than a floor.
    """
    assert vision_channel_staleness_pressure(0.0) == 0.0
    assert vision_channel_staleness_pressure(0.1) == 0.0
    assert vision_channel_staleness_pressure(5.0) == 0.0


def test_ordinary_jitter_inside_the_deadband_is_still_zero() -> None:
    # 8.6s is the measured live cadence of the slowest vision channel.
    assert vision_channel_staleness_pressure(8.6) == 0.0
    assert vision_channel_staleness_pressure(15.0) == 0.0


def test_pressure_rises_once_past_the_deadband() -> None:
    # Hand-computed: grace 15, saturation 60, span 45. At 37.5s the excess is
    # 22.5, which is half the span.
    assert vision_channel_staleness_pressure(37.5) == 0.5
    assert vision_channel_staleness_pressure(26.25) == 0.25


def test_saturates_at_one_and_never_exceeds_it() -> None:
    assert vision_channel_staleness_pressure(60.0) == 1.0
    assert vision_channel_staleness_pressure(3600.0) == 1.0
    assert vision_channel_staleness_pressure(1e9) == 1.0


def test_silence_converges_toward_alarm_not_toward_calm() -> None:
    """
    The property that makes this safe where a decay channel would not be: a
    longer silence must never read calmer. This is the node:substrate.route
    failure mode, asserted rather than commented.
    """
    ages = [0.0, 10.0, 20.0, 30.0, 45.0, 90.0, 86400.0]
    values = [vision_channel_staleness_pressure(a) for a in ages]
    assert values == sorted(values), values
    assert values[0] == 0.0 and values[-1] == 1.0


def test_non_finite_age_is_treated_as_calm_not_as_alarm() -> None:
    # Fail-open: a malformed clock reading must not fabricate an outage.
    assert vision_channel_staleness_pressure(float("nan")) == 0.0
    assert vision_channel_staleness_pressure(float("inf")) == 0.0


def test_negative_age_is_clamped_to_calm() -> None:
    # Clock skew between this host and the bus writer.
    assert vision_channel_staleness_pressure(-30.0) == 0.0


def test_degenerate_window_saturates_immediately() -> None:
    assert vision_channel_staleness_pressure(
        20.0, grace_seconds=30.0, saturation_seconds=10.0
    ) == 0.0
    assert vision_channel_staleness_pressure(
        40.0, grace_seconds=30.0, saturation_seconds=30.0
    ) == 1.0


def test_a_blinded_eye_is_invisible_to_availability_alone() -> None:
    """
    The failure that motivated the yield channel, as an executable assertion.

    A capped lens or dark room keeps artifacts flowing on schedule, so
    availability reads a clean 0.0 while every frame is empty. Verified against
    the real detector: a black probe frame returned ok=True with 0 objects
    where the live scene returned 6.
    """
    assert vision_channel_staleness_pressure(5.0) == 0.0  # artifacts on time
    assert perceptual_yield([0] * 60) == 0.0              # and carrying nothing
    assert perceptual_blindness_pressure([0] * 60) == 1.0


def test_yield_is_a_raw_observable_not_a_pressure() -> None:
    # No normalisation: 6.75 means 6.75 objects per frame, not a 0-1 score.
    assert perceptual_yield([6, 8, 7, 6]) == 6.75
    assert perceptual_yield([]) == 0.0


def test_blindness_needs_sustained_evidence_not_one_empty_frame() -> None:
    # A single empty frame is a blink; the guard prevents a cold-start alarm.
    assert perceptual_blindness_pressure([0]) == 0.0
    assert perceptual_blindness_pressure([0, 0, 0]) == 0.0
    assert perceptual_blindness_pressure([0] * 12) == 1.0


def test_any_real_detection_clears_blindness() -> None:
    assert perceptual_blindness_pressure([0] * 59 + [1]) == 0.0


def test_negative_counts_are_clamped_not_trusted() -> None:
    assert perceptual_yield([-5, 5]) == 2.5
