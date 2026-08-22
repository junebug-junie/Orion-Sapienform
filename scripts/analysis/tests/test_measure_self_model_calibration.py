"""Deterministic unit tests for measure_self_model_calibration.py.

No DB -- exercises the pure parsing/sampling/binning/statistics layer with
synthetic fixtures, same dynamic-module-load convention as
test_measure_ast_hot_reducer.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_self_model_calibration.py"
_spec = importlib.util.spec_from_file_location("measure_self_model_calibration", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_self_model_calibration"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)


def _row(i: int, *, predicted_shift, confidence, biometrics):
    return mod.SelfModelRow(
        generated_at=BASE + timedelta(seconds=30 * i),
        predicted_shift=predicted_shift,
        prediction_error_confidence=confidence,
        prediction_error_by_domain={"biometrics": biometrics},
    )


# --- parse_predicted_shift ---------------------------------------------------


def test_parse_predicted_shift_rising():
    assert mod.parse_predicted_shift(
        "biometrics prediction-error rising (trend=+0.1234 over recent window)"
    ) == ("biometrics", 1)


def test_parse_predicted_shift_falling():
    assert mod.parse_predicted_shift(
        "execution prediction-error falling (trend=-0.5000 over recent window)"
    ) == ("execution", -1)


def test_parse_predicted_shift_none():
    assert mod.parse_predicted_shift(None) is None
    assert mod.parse_predicted_shift("") is None
    assert mod.parse_predicted_shift("garbage") is None


# --- build_calibration_samples ----------------------------------------------


def test_correct_prediction_is_scored_correct():
    rows = [
        _row(0, predicted_shift="biometrics prediction-error rising (trend=+0.1)", confidence=0.9, biometrics=0.1),
        _row(1, predicted_shift=None, confidence=None, biometrics=0.1),
        _row(2, predicted_shift=None, confidence=None, biometrics=0.5),  # rose -> correct
    ]
    samples, n_no_shift, n_missing, n_ambig = mod.build_calibration_samples(rows, horizon_ticks=2)
    assert len(samples) == 1
    assert samples[0].correct is True
    assert samples[0].domain == "biometrics"
    assert samples[0].predicted_direction == 1
    assert n_no_shift == 2  # rows 1 and 2 have no predicted_shift/confidence of their own


def test_incorrect_prediction_is_scored_incorrect():
    rows = [
        _row(0, predicted_shift="biometrics prediction-error rising (trend=+0.1)", confidence=0.9, biometrics=0.5),
        _row(1, predicted_shift=None, confidence=None, biometrics=0.5),
        _row(2, predicted_shift=None, confidence=None, biometrics=0.1),  # fell -> incorrect
    ]
    samples, *_ = mod.build_calibration_samples(rows, horizon_ticks=2)
    assert len(samples) == 1
    assert samples[0].correct is False


def test_zero_delta_is_ambiguous_not_counted_either_way():
    rows = [
        _row(0, predicted_shift="biometrics prediction-error rising (trend=+0.1)", confidence=0.9, biometrics=0.3),
        _row(1, predicted_shift=None, confidence=None, biometrics=0.3),
        _row(2, predicted_shift=None, confidence=None, biometrics=0.3),  # unchanged
    ]
    samples, n_no_shift, n_missing, n_ambiguous = mod.build_calibration_samples(rows, horizon_ticks=2)
    assert samples == []
    assert n_ambiguous == 1


def test_missing_domain_at_horizon_is_skipped_honestly():
    rows = [
        _row(0, predicted_shift="chat prediction-error rising (trend=+0.1)", confidence=0.9, biometrics=0.3),
        _row(1, predicted_shift=None, confidence=None, biometrics=0.3),
        _row(2, predicted_shift=None, confidence=None, biometrics=0.3),
    ]
    # predicted domain is "chat" but rows only carry "biometrics" -> missing
    samples, n_no_shift, n_domain_missing, n_ambiguous = mod.build_calibration_samples(rows, horizon_ticks=2)
    assert samples == []
    assert n_domain_missing == 1


def test_row_too_close_to_end_of_window_is_skipped():
    rows = [
        _row(0, predicted_shift="biometrics prediction-error rising (trend=+0.1)", confidence=0.9, biometrics=0.3),
    ]
    samples, n_no_shift, n_domain_missing, n_ambiguous = mod.build_calibration_samples(rows, horizon_ticks=2)
    assert samples == []
    assert n_domain_missing == 1


# --- chronological_split -----------------------------------------------------


def test_chronological_split_no_shuffle():
    samples = [
        mod.CalibrationSample(BASE, "biometrics", 1, 0.5, True),
        mod.CalibrationSample(BASE, "biometrics", 1, 0.6, False),
        mod.CalibrationSample(BASE, "biometrics", 1, 0.7, True),
        mod.CalibrationSample(BASE, "biometrics", 1, 0.8, True),
    ]
    train, test = mod.chronological_split(samples, train_fraction=0.5)
    assert train == samples[:2]
    assert test == samples[2:]


# --- binning + stats ----------------------------------------------------------


def test_train_bin_edges_and_assign_bin_roundtrip():
    train_confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    edges = mod.train_bin_edges(train_confidences, n_bins=2)
    assert len(edges) == 1
    # low value goes to bin 0, high value goes to bin 1
    assert mod.assign_bin(0.1, edges) == 0
    assert mod.assign_bin(1.0, edges) == 1


def test_bin_accuracy_counts_correctly():
    edges = [0.5]
    test_samples = [
        mod.CalibrationSample(BASE, "biometrics", 1, 0.2, True),
        mod.CalibrationSample(BASE, "biometrics", 1, 0.3, False),
        mod.CalibrationSample(BASE, "biometrics", 1, 0.8, True),
        mod.CalibrationSample(BASE, "biometrics", 1, 0.9, True),
    ]
    stats = mod.bin_accuracy(test_samples, edges=edges, n_bins=2)
    assert stats[0].n == 2 and stats[0].n_correct == 1
    assert stats[1].n == 2 and stats[1].n_correct == 2
    assert stats[1].accuracy == 1.0


def test_pearson_correlation_perfect_positive():
    r = mod.pearson_correlation([0.1, 0.2, 0.3, 0.4], [0.0, 1.0, 2.0, 3.0])
    assert r is not None and abs(r - 1.0) < 1e-9


def test_pearson_correlation_undefined_on_zero_variance():
    assert mod.pearson_correlation([0.5, 0.5, 0.5], [0.0, 1.0, 0.0]) is None
    assert mod.pearson_correlation([0.1], [0.2]) is None


def test_two_proportion_z_matches_hand_computation():
    # bottom bin: 5/10 correct, top bin: 9/10 correct -- top should score higher
    z = mod.two_proportion_z(10, 5, 10, 9)
    assert z is not None and z < 0  # (n1,x1) - (n2,x2) with n1 worse -> negative
    z_flipped = mod.two_proportion_z(10, 9, 10, 5)
    assert z_flipped is not None and z_flipped > 0
    assert abs(z + z_flipped) < 1e-9


def test_two_proportion_z_none_when_empty():
    assert mod.two_proportion_z(0, 0, 10, 5) is None
