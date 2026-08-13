"""Unit tests for the vision-scoped field signals feeding capability:vision.

Two channels, deliberately separate: an irregularity channel that reuses the
mesh-wide counting rule, and an availability channel that exists because the
first one structurally cannot see silence.
"""

from __future__ import annotations

import math

from orion.substrate.prediction_error import (
    bus_synaptic_prediction_error,
    vision_channel_prediction_error,
    vision_channel_staleness_pressure,
)


# --- irregularity channel -------------------------------------------------


def test_matches_the_mesh_wide_counting_rule_exactly() -> None:
    # It delegates on purpose; this pins that it stays a delegation rather than
    # drifting into a second, separately-calibrated formula.
    for zs in ([], [0.0], [3.0], [0.5, 4.2, -7.0], [-3.0, 2.99]):
        assert vision_channel_prediction_error(zs) == bus_synaptic_prediction_error(zs)


def test_empty_edge_list_is_zero() -> None:
    assert vision_channel_prediction_error([]) == 0.0


def test_counts_anomalous_fraction_at_the_three_sigma_convention() -> None:
    # 2 of 10 anomalous -- the exact reading observed live at 05:20:04 on
    # 2026-08-13 while the mesh-wide value sat at 0.0175.
    zscores = [3.4, -5.1] + [0.2] * 8
    assert vision_channel_prediction_error(zscores) == 0.2


def test_boundary_is_inclusive_at_three_sigma() -> None:
    assert vision_channel_prediction_error([3.0]) == 1.0
    assert vision_channel_prediction_error([2.999]) == 0.0


def test_one_enormous_outlier_counts_the_same_as_one_marginal_one() -> None:
    # The heavy-tail failure that pinned the mesh-wide node at 1.0 cannot recur.
    assert vision_channel_prediction_error([7087.8, 0.1, 0.1, 0.1]) == 0.25
    assert vision_channel_prediction_error([3.01, 0.1, 0.1, 0.1]) == 0.25


def test_negative_zscores_count_by_magnitude() -> None:
    assert vision_channel_prediction_error([-4.0, 0.1]) == 0.5


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


def test_a_frozen_anomaly_reading_does_not_mask_an_outage() -> None:
    """
    The design-invalidating case, as an executable assertion.

    compute_ewma_update runs per observed message, so a silent channel freezes
    its gap_zscore at whatever it held while healthy -- normally calm. If the
    two channels were not combined, a dead camera would read as healthy.
    """
    frozen_calm_zscores = [0.1, -0.2, 0.05]
    assert vision_channel_prediction_error(frozen_calm_zscores) == 0.0

    dead_for_two_minutes = vision_channel_staleness_pressure(120.0)
    assert dead_for_two_minutes == 1.0

    # capability:vision maps both to `pressure`, combined by apply_diffusion's max().
    assert max(vision_channel_prediction_error(frozen_calm_zscores), dead_for_two_minutes) == 1.0
