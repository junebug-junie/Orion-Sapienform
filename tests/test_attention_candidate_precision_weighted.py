"""Unit tests for Candidate A (`orion/attention/field_attention/
candidate_precision_weighted.py`) -- precision-weighted prediction-error salience,
Feldman & Friston 2010. Shadow-only pure function; see the module docstring and
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-
plan.md` for the design record. No I/O, no fixtures needed."""

from __future__ import annotations

import pytest

from orion.attention.field_attention.candidate_precision_weighted import (
    PRECISION_VARIANCE_FLOOR,
    normalize_across_targets,
    precision_weighted_salience,
)


def test_empty_history_yields_zero_everything() -> None:
    result = precision_weighted_salience([])
    assert result.salience == 0.0
    assert result.precision == 0.0
    assert result.variance == 0.0
    assert result.current_error == 0.0
    assert result.n_samples == 0
    assert result.variance_floored is False


def test_single_sample_history_floors_variance_and_uses_that_sample_as_current() -> None:
    """One real observation has zero variance by definition -- treated as the
    smallest-possible-sample-size floor case, not a separate undefined case."""
    result = precision_weighted_salience([0.05])
    assert result.n_samples == 1
    assert result.current_error == pytest.approx(0.05)
    assert result.variance == 0.0
    assert result.variance_floored is True
    assert result.precision == pytest.approx(1.0 / PRECISION_VARIANCE_FLOOR)
    assert result.salience == pytest.approx(0.05 / PRECISION_VARIANCE_FLOOR)


def test_near_zero_variance_edge_case_floors_precision_not_diverges() -> None:
    """A target whose error has been almost perfectly constant -- the concrete
    'variance-near-zero instability risk' named in the design doc. Must not raise
    ZeroDivisionError or produce +inf."""
    history = [0.03, 0.03, 0.03, 0.03, 0.0300001]
    result = precision_weighted_salience(history)
    assert result.n_samples == 5
    assert result.variance == pytest.approx(0.0, abs=1e-9)
    assert result.variance_floored is True
    assert result.precision == pytest.approx(1.0 / PRECISION_VARIANCE_FLOOR)
    assert result.salience == pytest.approx(
        (1.0 / PRECISION_VARIANCE_FLOOR) * abs(0.0300001)
    )
    import math

    assert math.isfinite(result.precision)
    assert math.isfinite(result.salience)


def test_healthy_variance_produces_finite_bounded_precision() -> None:
    """A real, non-degenerate history (modeled on the live biometrics spread found
    2026-07-21: real values roughly in [0.0, 0.17]) should NOT trip the variance
    floor, and should produce a precision that is large but not floor-ceiling-pinned."""
    history = [0.001, 0.08, 0.003, 0.12, 0.0005, 0.09, 0.002, 0.15]
    result = precision_weighted_salience(history)
    assert result.n_samples == 8
    assert result.variance > PRECISION_VARIANCE_FLOOR
    assert result.variance_floored is False
    assert result.current_error == pytest.approx(0.15)
    expected_precision = 1.0 / result.variance
    assert result.precision == pytest.approx(expected_precision)
    assert result.salience == pytest.approx(expected_precision * 0.15)
    # Sanity: precision should be well below the floor-ceiling for a healthy series.
    assert result.precision < 1.0 / PRECISION_VARIANCE_FLOOR


def test_current_error_is_the_last_element_not_the_max_or_mean() -> None:
    history = [0.5, 0.5, 0.01]
    result = precision_weighted_salience(history)
    assert result.current_error == pytest.approx(0.01)


def test_zero_current_error_yields_zero_salience_regardless_of_precision() -> None:
    history = [0.02, 0.03, 0.0]
    result = precision_weighted_salience(history)
    assert result.current_error == 0.0
    assert result.salience == 0.0
    assert result.precision > 0.0  # precision itself is unaffected by the current value


def test_negative_values_are_handled_via_absolute_value_of_current_error() -> None:
    """prediction_error instruments in orion/substrate/prediction_error.py are all
    non-negative by construction (min(1.0, mean/threshold) or a [0,1] mismatch rate),
    but this function does not assume that -- abs() is applied defensively."""
    history = [0.1, -0.2, 0.15, -0.3]
    result = precision_weighted_salience(history)
    assert result.current_error == pytest.approx(-0.3)
    assert result.salience == pytest.approx(result.precision * 0.3)
    assert result.salience >= 0.0


def test_result_is_a_frozen_dataclass_not_mutable() -> None:
    result = precision_weighted_salience([0.1, 0.2])
    with pytest.raises(Exception):
        result.salience = 999.0  # type: ignore[misc]


# -- normalize_across_targets --------------------------------------------------
# Added in review (2026-07-22): precision_weighted_salience()'s raw output is
# unbounded and dominated by each target's own historical variance scale, which
# is not a valid drop-in for FieldAttentionTargetV1.salience_score (schema-bound
# to [0,1]) and is not meaningfully comparable across targets without this step.


def test_normalize_across_targets_empty_input() -> None:
    assert normalize_across_targets({}) == {}


def test_normalize_across_targets_maps_min_to_zero_and_max_to_one() -> None:
    result = normalize_across_targets({"a": 10.0, "b": 400.0, "c": 57100.0})
    assert result["c"] == pytest.approx(1.0)
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx((400.0 - 10.0) / (57100.0 - 10.0))


def test_normalize_across_targets_preserves_relative_rank() -> None:
    raw = {"low": 5.0, "mid": 500.0, "high": 50000.0}
    result = normalize_across_targets(raw)
    assert result["low"] < result["mid"] < result["high"]


def test_normalize_across_targets_output_always_in_unit_interval() -> None:
    raw = {"a": 0.0, "b": 3.3, "c": 1e6, "d": 42.0}
    result = normalize_across_targets(raw)
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_normalize_across_targets_all_equal_scores_get_one_not_zero() -> None:
    """A real tie must not be misrepresented as 'nothing here matters' -- there is
    no basis to floor a genuine tie to 0.0."""
    result = normalize_across_targets({"a": 42.0, "b": 42.0, "c": 42.0})
    assert result == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_normalize_across_targets_single_target_gets_one() -> None:
    """Degenerate case of the tie rule: the only real competitor this tick gets
    maximal (not zero, not arbitrary) attention -- it's the only real candidate."""
    assert normalize_across_targets({"only": 12345.6}) == {"only": 1.0}


def test_normalize_across_targets_near_equal_scores_within_epsilon_treated_as_tie() -> None:
    result = normalize_across_targets({"a": 1.0, "b": 1.0 + 1e-13})
    assert result == {"a": 1.0, "b": 1.0}


def test_normalize_across_targets_does_not_mutate_input() -> None:
    raw = {"a": 1.0, "b": 2.0}
    normalize_across_targets(raw)
    assert raw == {"a": 1.0, "b": 2.0}


def test_normalize_across_targets_end_to_end_with_real_precision_weighted_salience() -> None:
    """Integration-shaped: run two targets through the real
    precision_weighted_salience() pure function, then normalize the raw results --
    exercises both functions together the way a real caller would, not in isolation."""
    quiet_target = precision_weighted_salience([0.03, 0.03, 0.03, 0.03, 0.031])
    noisy_target = precision_weighted_salience([0.01, 0.08, 0.02, 0.12, 0.15])
    raw = {"quiet": quiet_target.salience, "noisy": noisy_target.salience}
    normalized = normalize_across_targets(raw)
    assert set(normalized) == {"quiet", "noisy"}
    assert all(0.0 <= v <= 1.0 for v in normalized.values())
