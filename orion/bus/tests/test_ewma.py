from __future__ import annotations

import math

import pytest

from orion.bus.ewma import compute_ewma_update


def test_first_observation_has_no_zscore() -> None:
    update = compute_ewma_update(prev_ewma=0.0, prev_variance=0.0, prev_count=0, value=5.0, alpha=0.2)
    assert update.zscore is None
    assert update.ewma == 5.0
    assert update.variance == 0.0


def test_second_observation_computes_zscore_against_prior_baseline() -> None:
    update = compute_ewma_update(prev_ewma=5.0, prev_variance=1.0, prev_count=1, value=5.0, alpha=0.2)
    assert update.zscore == pytest.approx(0.0)


def test_large_deviation_produces_large_zscore() -> None:
    update = compute_ewma_update(prev_ewma=5.0, prev_variance=1.0, prev_count=1, value=50.0, alpha=0.2)
    assert update.zscore is not None
    assert update.zscore > 10.0


def test_zero_variance_does_not_divide_by_zero() -> None:
    update = compute_ewma_update(prev_ewma=5.0, prev_variance=0.0, prev_count=1, value=5.5, alpha=0.2)
    assert update.zscore is not None
    assert math.isfinite(update.zscore)


def test_ewma_moves_toward_new_value() -> None:
    update = compute_ewma_update(prev_ewma=10.0, prev_variance=1.0, prev_count=5, value=20.0, alpha=0.5)
    assert 10.0 < update.ewma < 20.0
