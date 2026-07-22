"""Deterministic unit tests for measure_transport_biometrics_prediction_error_correlation.py.

No DB. All pure functions operate on synthetic float series.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "measure_transport_biometrics_prediction_error_correlation.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_transport_biometrics_prediction_error_correlation", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_transport_biometrics_prediction_error_correlation"] = mod
_spec.loader.exec_module(mod)


def test_pearson_correlation_perfect_positive() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert mod.pearson_correlation(x, y) == pytest.approx(1.0)


def test_pearson_correlation_none_when_x_constant() -> None:
    x = [1.0, 1.0, 1.0, 1.0]
    y = [1.0, 2.0, 3.0, 4.0]
    assert mod.pearson_correlation(x, y) is None


def test_pearson_correlation_none_when_both_constant() -> None:
    assert mod.pearson_correlation([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) is None


def test_pearson_correlation_none_for_fewer_than_two_points() -> None:
    assert mod.pearson_correlation([1.0], [2.0]) is None


def test_pearson_correlation_none_for_mismatched_lengths() -> None:
    assert mod.pearson_correlation([1.0, 2.0], [1.0]) is None


def test_compute_correlation_degenerate_transport_reports_which_series() -> None:
    """Matches the real, currently-observed shape: transport flat, biometrics real."""
    transport = [0.0] * 50
    biometrics = [0.01 * i for i in range(50)]
    result = mod.compute_correlation(transport, biometrics)
    assert result.correlation is None
    assert result.transport_variance == 0.0
    assert result.biometrics_variance > 0.0
    assert "transport" in result.verdict


def test_compute_correlation_both_degenerate() -> None:
    result = mod.compute_correlation([0.0] * 10, [0.0] * 10)
    assert result.correlation is None
    assert "transport" in result.verdict
    assert "biometrics" in result.verdict


def test_compute_correlation_insufficient_data() -> None:
    result = mod.compute_correlation([0.1], [0.2])
    assert result.n == 1
    assert "INSUFFICIENT DATA" in result.verdict


def test_compute_correlation_real_correlation_found() -> None:
    transport = [0.01 * i for i in range(30)]
    biometrics = [0.02 * i for i in range(30)]
    result = mod.compute_correlation(transport, biometrics)
    assert result.correlation == pytest.approx(1.0)
    assert "Real, non-trivial correlation found" in result.verdict


def test_compute_correlation_truncates_to_shorter_series() -> None:
    transport = [0.01 * i for i in range(10)]
    biometrics = [0.02 * i for i in range(30)]
    result = mod.compute_correlation(transport, biometrics)
    assert result.n == 10
