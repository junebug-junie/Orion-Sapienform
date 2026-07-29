"""Deterministic unit tests for measure_attention_self_model_confidence_baseline.py.

No DB. All functions under test are pure -- exercised directly with synthetic
ConfidenceTick sequences, same dynamic-module-loading pattern as
scripts/analysis/tests/test_measure_ast_hot_reducer.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "measure_attention_self_model_confidence_baseline.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_attention_self_model_confidence_baseline", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_attention_self_model_confidence_baseline"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 7, 29, 0, 0, 0, tzinfo=UTC)


def _tick(offset_sec: float, pe_confidence, confidence=None, reason="field_salience_only"):
    return mod.ConfidenceTick(
        generated_at=BASE + timedelta(seconds=offset_sec),
        confidence=confidence if confidence is not None else pe_confidence,
        prediction_error_confidence=pe_confidence,
        attention_reason=reason,
        predicted_shift=None,
    )


def test_parse_self_model_row_extracts_real_fields():
    payload = {
        "confidence": 0.7,
        "prediction_error_confidence": 0.85,
        "attention_reason": "bottom_up_salience",
        "predicted_shift": "biometrics prediction-error rising",
    }
    tick = mod.parse_self_model_row(payload, BASE)
    assert tick.confidence == 0.7
    assert tick.prediction_error_confidence == 0.85
    assert tick.attention_reason == "bottom_up_salience"
    assert tick.predicted_shift == "biometrics prediction-error rising"


def test_parse_self_model_row_handles_missing_fields_honestly():
    tick = mod.parse_self_model_row({}, BASE)
    assert tick.confidence is None
    assert tick.prediction_error_confidence is None
    assert tick.attention_reason == "no_data"


def test_parse_self_model_row_rejects_non_dict_payload():
    assert mod.parse_self_model_row(None, BASE) is None
    assert mod.parse_self_model_row("not a dict", BASE) is None


def test_degenerate_check_flags_permanent_ceiling():
    values = [0.999] * 100
    result = mod.degenerate_check(values)
    assert "SUSPICIOUS" in result["verdict"]
    assert "ceiling" in result["verdict"]


def test_degenerate_check_flags_permanent_floor():
    values = [0.001] * 100
    result = mod.degenerate_check(values)
    assert "SUSPICIOUS" in result["verdict"]
    assert "floor" in result["verdict"]


def test_degenerate_check_flags_zero_variance():
    values = [0.5] * 50
    result = mod.degenerate_check(values)
    assert "SUSPICIOUS" in result["verdict"]


def test_degenerate_check_ok_for_real_variance():
    values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.2, 0.8, 0.3]
    result = mod.degenerate_check(values)
    assert result["verdict"] == "OK"
    assert result["min"] == 0.1
    assert result["max"] == 0.9


def test_degenerate_check_empty_input():
    result = mod.degenerate_check([])
    assert result["verdict"] == "NO_DATA"
    assert result["count"] == 0


def test_detect_recovery_events_finds_discrete_single_tick_jump():
    """Low -> immediately high next tick is the discrete-event signature."""
    ticks = [
        _tick(0, 0.6),  # above low_threshold -- not yet armed
        _tick(30, 0.2),  # arms here (index 1)
        _tick(60, 0.9),  # sharp jump -- fires 1 tick after arming
        _tick(90, 0.85),
    ]
    events = mod.detect_recovery_events(ticks, low_threshold=0.5, high_threshold=0.8)
    assert len(events) == 1
    assert events[0].ticks_to_cross == 1
    assert events[0].low_value == 0.2
    assert events[0].high_value == 0.9


def test_detect_recovery_events_finds_smooth_gradual_climb():
    ticks = [
        _tick(0, 0.2),
        _tick(30, 0.35),
        _tick(60, 0.5),
        _tick(90, 0.65),
        _tick(120, 0.85),  # crosses high after 4 ticks of gradual climb
    ]
    events = mod.detect_recovery_events(ticks, low_threshold=0.5, high_threshold=0.8)
    assert len(events) == 1
    assert events[0].ticks_to_cross == 4


def test_detect_recovery_events_does_not_fire_on_small_wiggle_above_low():
    """Never actually drops to/below low_threshold -- must not fire."""
    ticks = [_tick(0, 0.6), _tick(30, 0.55), _tick(60, 0.9)]
    events = mod.detect_recovery_events(ticks, low_threshold=0.5, high_threshold=0.8)
    assert events == []


def test_detect_recovery_events_requires_rearm_between_events():
    """Two low->high crossings, correctly counted as two separate events, not one."""
    ticks = [
        _tick(0, 0.2),
        _tick(30, 0.9),  # event 1
        _tick(60, 0.85),
        _tick(90, 0.3),  # drops back down, re-arms
        _tick(120, 0.95),  # event 2
    ]
    events = mod.detect_recovery_events(ticks, low_threshold=0.5, high_threshold=0.8)
    assert len(events) == 2


def test_detect_recovery_events_ignores_none_values():
    ticks = [
        mod.ConfidenceTick(BASE, None, None, "no_data", None),
        _tick(30, 0.2),
        _tick(60, 0.9),
    ]
    events = mod.detect_recovery_events(ticks, low_threshold=0.5, high_threshold=0.8)
    assert len(events) == 1


def test_delta_stats_high_near_zero_fraction_for_mostly_flat_signal():
    ticks = [_tick(i * 30, 0.5 + (0.01 if i % 2 else 0.0)) for i in range(10)]
    stats = mod.delta_stats(ticks, field="prediction_error_confidence")
    assert stats["frac_near_zero"] > 0.5


def test_delta_stats_empty_for_insufficient_data():
    stats = mod.delta_stats([_tick(0, 0.5)], field="prediction_error_confidence")
    assert stats["count"] == 0


def test_render_report_flags_insufficient_history_when_no_ticks():
    report = mod.render_report(since_hours=6.0, ticks=[], low_threshold=0.5, high_threshold=0.8)
    assert "INSUFFICIENT HISTORY" in report


def test_render_report_flags_not_yet_trustable_below_minimum_window():
    ticks = [_tick(i * 30, 0.3 + 0.01 * i) for i in range(5)]
    report = mod.render_report(since_hours=6.0, ticks=ticks, low_threshold=0.5, high_threshold=0.8)
    assert "NOT YET TRUSTABLE" in report


def test_render_report_verdict_discrete_for_sharp_crossings():
    # Enough ticks + enough real span to clear the trusted-verdict floor, all
    # crossings single-tick.
    ticks = []
    t = 0.0
    for _ in range(80):
        ticks.append(_tick(t, 0.2))
        t += 150
        ticks.append(_tick(t, 0.9))
        t += 150
        ticks.append(_tick(t, 0.2))
        t += 150
    report = mod.render_report(since_hours=6.0, ticks=ticks, low_threshold=0.5, high_threshold=0.8)
    assert "discrete events" in report


def test_render_report_verdict_smooth_for_gradual_crossings():
    """Review finding 2026-07-29: the smooth-verdict branch (the more
    consequential half of this script's whole purpose) had zero coverage.
    Each crossing here climbs gradually over several ticks -- median
    ticks_to_cross must land above the discrete/smooth cutoff (2)."""
    ticks = []
    t = 0.0
    for _ in range(60):
        for value in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.2):
            ticks.append(_tick(t, value))
            t += 150
    report = mod.render_report(since_hours=6.0, ticks=ticks, low_threshold=0.5, high_threshold=0.8)
    assert "leans smooth, not discrete" in report


def test_render_report_verdict_no_events_in_trustable_window():
    """Middle branch: window clears the trust floor but the signal never
    actually crosses low_threshold, so zero events fire."""
    ticks = [_tick(i * 40, 0.85) for i in range(210)]  # 210 * 40s = 2.33h, all above low
    report = mod.render_report(since_hours=6.0, ticks=ticks, low_threshold=0.5, high_threshold=0.8)
    assert "No low->high recovery events observed" in report


def test_render_report_flags_truncation():
    ticks = [_tick(i * 30, 0.5) for i in range(5)]
    report = mod.render_report(
        since_hours=6.0, ticks=ticks, low_threshold=0.5, high_threshold=0.8, truncated=True
    )
    assert "TRUNCATED" in report


def test_render_report_no_truncation_note_by_default():
    ticks = [_tick(i * 30, 0.5) for i in range(5)]
    report = mod.render_report(since_hours=6.0, ticks=ticks, low_threshold=0.5, high_threshold=0.8)
    assert "TRUNCATED" not in report


def test_detect_recovery_events_boundary_ties_are_inclusive():
    """Docstring claims <=/>= inclusive semantics -- pin it down explicitly."""
    ticks = [_tick(0, 0.6), _tick(30, 0.5), _tick(60, 0.8)]  # exact ties on both bands
    events = mod.detect_recovery_events(ticks, low_threshold=0.5, high_threshold=0.8)
    assert len(events) == 1


def test_delta_stats_and_detect_recovery_events_respect_field_selector():
    """Review finding 2026-07-29: every prior test used a fixture where
    confidence == prediction_error_confidence, so a copy-paste bug reading
    the wrong attribute internally would still have passed everything.
    Uses genuinely different values per field to prove the selector works."""
    ticks = [
        _tick(0, pe_confidence=0.9, confidence=0.1),
        _tick(30, pe_confidence=0.9, confidence=0.9),  # confidence: low->high jump
        _tick(60, pe_confidence=0.2, confidence=0.9),  # prediction_error_confidence: high->low
    ]
    confidence_events = mod.detect_recovery_events(ticks, field="confidence", low_threshold=0.5, high_threshold=0.8)
    pe_events = mod.detect_recovery_events(
        ticks, field="prediction_error_confidence", low_threshold=0.5, high_threshold=0.8
    )
    assert len(confidence_events) == 1  # confidence: 0.1 -> 0.9
    assert len(pe_events) == 0  # prediction_error_confidence never drops to/below 0.5

    confidence_deltas = mod.delta_stats(ticks, field="confidence")
    pe_deltas = mod.delta_stats(ticks, field="prediction_error_confidence")
    assert confidence_deltas["max_abs_delta"] != pe_deltas["max_abs_delta"]


def test_parse_self_model_row_rejects_boolean_confidence():
    """bool is a subclass of int in Python -- isinstance(True, (int, float))
    is True, so a stray boolean value would silently become 1.0/0.0 without
    this explicit rejection."""
    tick = mod.parse_self_model_row({"confidence": True, "prediction_error_confidence": False}, BASE)
    assert tick.confidence is None
    assert tick.prediction_error_confidence is None


def test_parse_self_model_row_rejects_nan_and_infinity():
    tick = mod.parse_self_model_row(
        {"confidence": float("nan"), "prediction_error_confidence": float("inf")}, BASE
    )
    assert tick.confidence is None
    assert tick.prediction_error_confidence is None


def test_write_ticks_csv_round_trips(tmp_path):
    import csv

    ticks = [_tick(0, 0.5, confidence=0.6, reason="bottom_up_salience")]
    out_path = tmp_path / "ticks.csv"
    mod.write_ticks_csv(out_path, ticks)
    with out_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["attention_reason"] == "bottom_up_salience"
    assert rows[0]["confidence"] == "0.6000"
    assert rows[0]["prediction_error_confidence"] == "0.5000"
