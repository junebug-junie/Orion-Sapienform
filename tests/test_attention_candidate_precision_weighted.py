"""Unit tests for Candidate A (`orion/attention/field_attention/
candidate_precision_weighted.py`) -- precision-weighted prediction-error salience,
Feldman & Friston 2010. Shadow-only pure function; see the module docstring and
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-
plan.md` for the design record. No I/O, no fixtures needed."""

from __future__ import annotations

import pytest

from orion.attention.field_attention.candidate_precision_weighted import (
    PRECISION_VARIANCE_FLOOR,
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
