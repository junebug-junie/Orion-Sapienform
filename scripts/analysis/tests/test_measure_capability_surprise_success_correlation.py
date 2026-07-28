"""Deterministic unit tests for measure_capability_surprise_success_correlation.py.

No DB. All pure functions operate on synthetic float series.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "measure_capability_surprise_success_correlation.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_capability_surprise_success_correlation", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_capability_surprise_success_correlation"] = mod
_spec.loader.exec_module(mod)


def test_pearson_correlation_perfect_positive() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert mod.pearson_correlation(x, y) == pytest.approx(1.0)


def test_pearson_correlation_none_when_x_constant() -> None:
    x = [1.0, 1.0, 1.0, 1.0]
    y = [1.0, 2.0, 3.0, 4.0]
    assert mod.pearson_correlation(x, y) is None


def test_pearson_correlation_none_for_fewer_than_two_points() -> None:
    assert mod.pearson_correlation([1.0], [2.0]) is None


def test_pearson_correlation_none_for_mismatched_lengths() -> None:
    assert mod.pearson_correlation([1.0, 2.0], [1.0]) is None


def test_compute_correlation_degenerate_success_reports_which_series() -> None:
    """All successes -- success has zero variance even though surprise is real."""
    surprise = [0.01 * i for i in range(50)]
    success = [1.0] * 50
    result = mod.compute_correlation("web.fetch.readonly", surprise, success)
    assert result.correlation is None
    assert result.surprise_variance > 0.0
    assert result.success_variance == 0.0
    assert "success" in result.verdict


def test_compute_correlation_insufficient_data() -> None:
    result = mod.compute_correlation("recall.query.readonly", [0.1], [1.0])
    assert result.n < 2
    assert "INSUFFICIENT DATA" in result.verdict


def test_compute_correlation_real_correlation_found() -> None:
    """Synthetic: high surprise consistently correlates with failure."""
    surprise = [0.05, 0.1, 0.9, 0.95, 0.08, 0.85, 0.12, 0.9]
    success = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    result = mod.compute_correlation("web.fetch.readonly", surprise, success)
    assert result.correlation is not None
    assert result.correlation < -0.3
    assert "non-trivial correlation" in result.verdict


def test_compute_correlation_truncates_to_min_length() -> None:
    result = mod.compute_correlation("web.fetch.readonly", [0.1, 0.2, 0.3], [1.0, 0.0])
    assert result.n == 2
