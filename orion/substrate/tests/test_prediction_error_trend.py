"""Direct unit coverage for orion/substrate/prediction_error_trend.py.

Extracted 2026-07-29 from scripts/analysis/measure_ast_hot_reducer.py (see
that module's own docstring for the full backtest validation this formula
already has). scripts/analysis/tests/test_measure_ast_hot_reducer.py
continues to exercise this same function via its dynamic module loader --
this file adds direct, package-local coverage now that it's an importable
module in its own right, used by both the offline replay and the live
`_attention_broadcast_tick()` self-model tick.
"""

from __future__ import annotations

from orion.substrate.prediction_error_trend import compute_prediction_error_trend


def test_empty_window_yields_empty_trend():
    assert compute_prediction_error_trend([]) == {}


def test_single_tick_window_yields_empty_trend():
    assert compute_prediction_error_trend([{"biometrics": 0.1}]) == {}


def test_rising_then_falling_yields_negative_trend():
    window = [{"biometrics": 0.1}, {"biometrics": 0.2}, {"biometrics": 0.8}, {"biometrics": 0.9}]
    trend = compute_prediction_error_trend(window)
    # prior_half mean=0.15, recent_half mean=0.85 -> prior - recent < 0
    assert trend["biometrics"] < 0


def test_falling_then_rising_yields_positive_trend():
    window = [{"biometrics": 0.9}, {"biometrics": 0.8}, {"biometrics": 0.2}, {"biometrics": 0.1}]
    trend = compute_prediction_error_trend(window)
    assert trend["biometrics"] > 0


def test_domain_missing_from_one_half_is_excluded():
    window = [{"biometrics": 0.1}, {"execution": 0.5}]
    trend = compute_prediction_error_trend(window)
    assert trend == {}


def test_domain_present_in_both_halves_of_multi_domain_window():
    window = [
        {"biometrics": 0.1, "execution": 0.05},
        {"biometrics": 0.2, "execution": 0.05},
        {"biometrics": 0.9},
        {"biometrics": 0.8},
    ]
    trend = compute_prediction_error_trend(window)
    assert "biometrics" in trend
    assert "execution" not in trend  # absent from the recent half entirely
