"""Deterministic unit tests for measure_attention_reason_branch_starvation.py.

No DB, no CSV file I/O -- exercises the pure `analyze_branch_starvation` function
directly with synthetic row dicts, same pattern as
scripts/analysis/tests/test_measure_ast_hot_reducer.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "measure_attention_reason_branch_starvation.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_attention_reason_branch_starvation", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_attention_reason_branch_starvation"] = mod
_spec.loader.exec_module(mod)


def _row(reason: str, present: str, stale: str) -> dict:
    return {
        "attention_reason": reason,
        "broadcast_lane_present": present,
        "broadcast_lane_stale": stale,
    }


def test_all_broadcast_fresh_bottom_up_wins_every_tick():
    rows = [_row("bottom_up_salience", "True", "False") for _ in range(10)]
    result = mod.analyze_branch_starvation(rows)
    assert result["total_ticks"] == 10
    assert result["broadcast_fresh_pct"] == 100.0
    assert result["field_salience_only_count"] == 0
    assert result["elif_ordering_contradictions"] == 0
    # Starvation requires field_salience_only to actually be rare in this
    # dataset; with zero field_salience_only ticks and 100% freshness, the
    # hypothesis is trivially confirmed.
    assert result["starvation_confirmed"] is True


def test_realistic_live_ratio_confirms_starvation():
    # Mirrors the real 2026-07-24 8h-window measurement: 9204 bottom_up_salience,
    # 4 field_salience_only, all broadcast-fresh ticks except the 4 that fell
    # back (present=False or stale=True for those 4).
    rows = [_row("bottom_up_salience", "True", "False") for _ in range(9204)]
    rows += [_row("field_salience_only", "False", "True") for _ in range(4)]
    result = mod.analyze_branch_starvation(rows)
    assert result["total_ticks"] == 9208
    assert result["field_salience_only_count"] == 4
    assert result["broadcast_fresh_ticks"] == 9204
    assert result["broadcast_fresh_pct"] > 99.0
    assert result["field_salience_only_pct"] < 1.0
    assert result["elif_ordering_contradictions"] == 0
    assert result["starvation_confirmed"] is True


def test_contradiction_detected_when_field_salience_only_on_fresh_broadcast():
    # A broadcast-fresh tick claiming field_salience_only should never happen
    # per the reducer's real elif order -- this simulates a data/logic
    # mismatch and confirms the script flags it instead of silently reporting
    # a clean starvation story.
    rows = [_row("bottom_up_salience", "True", "False") for _ in range(5)]
    rows.append(_row("field_salience_only", "True", "False"))
    result = mod.analyze_branch_starvation(rows)
    assert result["elif_ordering_contradictions"] == 1
    assert result["starvation_confirmed"] is False


def test_not_starved_when_broadcast_mostly_stale():
    # If the broadcast lane were mostly stale/absent, field_salience_only
    # would fire often -- this is NOT a starvation scenario, and the function
    # must say so rather than always confirming the hypothesis.
    rows = [_row("field_salience_only", "False", "True") for _ in range(8)]
    rows += [_row("bottom_up_salience", "True", "False") for _ in range(2)]
    result = mod.analyze_branch_starvation(rows)
    assert result["broadcast_fresh_pct"] == 20.0
    assert result["starvation_confirmed"] is False


def test_malformed_bool_excluded_not_defaulted():
    rows = [
        _row("bottom_up_salience", "True", "False"),
        _row("no_data", "", ""),  # unparseable -- must be excluded, not defaulted
        _row("bottom_up_salience", "True", "False"),
    ]
    result = mod.analyze_branch_starvation(rows)
    assert result["total_ticks"] == 3
    assert result["broadcast_known_ticks"] == 2
    assert result["broadcast_fresh_ticks"] == 2


def test_field_salience_only_pct_gate_checked_independently():
    # Isolates the `field_salience_only_pct < 10.0` gate from the freshness and
    # contradiction gates: 85 known-fresh bottom_up_salience ticks (100% fresh
    # among known) plus 15 field_salience_only ticks with unparseable broadcast
    # bools (excluded from the known/fresh denominator entirely, so they can't
    # trip the contradiction check either). Freshness and contradiction gates
    # both pass; only the field_salience_only_pct gate (15% of the 100 total
    # ticks) should fail starvation_confirmed.
    rows = [_row("bottom_up_salience", "True", "False") for _ in range(85)]
    rows += [_row("field_salience_only", "", "") for _ in range(15)]
    result = mod.analyze_branch_starvation(rows)
    assert result["total_ticks"] == 100
    assert result["broadcast_known_ticks"] == 85
    assert result["broadcast_fresh_pct"] == 100.0
    assert result["elif_ordering_contradictions"] == 0
    assert result["field_salience_only_pct"] == 15.0
    assert result["starvation_confirmed"] is False


def test_empty_rows_no_crash():
    result = mod.analyze_branch_starvation([])
    assert result["total_ticks"] == 0
    assert result["broadcast_fresh_pct"] == 0.0
    assert result["starvation_confirmed"] is False


def test_render_report_smoke():
    rows = [_row("bottom_up_salience", "True", "False") for _ in range(3)]
    rows.append(_row("field_salience_only", "False", "True"))
    result = mod.analyze_branch_starvation(rows)
    report = mod.render_report(result, "/tmp/fake/ticks.csv")
    assert "attention_reason distribution" in report
    assert "Conclusion" in report
    assert "/tmp/fake/ticks.csv" in report
