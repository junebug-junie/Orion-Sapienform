"""Deterministic unit tests for measure_chat_attention_ground_truth_gap.py.

No DB, no network. Same sibling-module-by-file-path loading pattern as
test_measure_candidate_a_vs_b_head_to_head.py -- lives in top-level `tests/`,
not `scripts/analysis/tests/`, since `scripts` is in pyproject.toml's
`norecursedirs`.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "measure_chat_attention_ground_truth_gap.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_chat_attention_ground_truth_gap", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_chat_attention_ground_truth_gap"] = mod
_spec.loader.exec_module(mod)


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 30, hour, minute, tzinfo=timezone.utc)


# ===========================================================================
# compute_target_universe_disjointness
# ===========================================================================


def test_disjointness_zero_overlap() -> None:
    result = mod.compute_target_universe_disjointness(
        theme_keys=["open-loop-aaa", "open-loop-bbb"],
        target_ids=["node:athena", "capability:storage"],
    )
    assert result.theme_key_count == 2
    assert result.target_id_count == 2
    assert result.overlap == frozenset()
    assert result.is_disjoint is True


def test_disjointness_real_overlap_detected() -> None:
    result = mod.compute_target_universe_disjointness(
        theme_keys=["open-loop-aaa", "node:athena"],
        target_ids=["node:athena", "capability:storage"],
    )
    assert result.overlap == frozenset({"node:athena"})
    assert result.is_disjoint is False


def test_disjointness_drops_blank_and_none_values() -> None:
    result = mod.compute_target_universe_disjointness(
        theme_keys=["open-loop-aaa", "", None],  # type: ignore[list-item]
        target_ids=["node:athena", ""],
    )
    assert result.theme_key_count == 1
    assert result.target_id_count == 1


def test_disjointness_empty_inputs() -> None:
    result = mod.compute_target_universe_disjointness([], [])
    assert result.theme_key_count == 0
    assert result.target_id_count == 0
    assert result.is_disjoint is True


def test_disjointness_empty_universe_is_inconclusive_not_disjoint() -> None:
    # Code-review regression test: a 0-overlap result computed from an empty
    # side (e.g. a fetch failure on one table) must not be reported as a
    # real "DISJOINT" finding -- that's the metric-quality-gate's
    # "suspiciously clean 0.0 read as confirmed calm" failure class.
    empty_target_side = mod.compute_target_universe_disjointness(
        theme_keys=["open-loop-aaa"], target_ids=[]
    )
    assert empty_target_side.is_disjoint is True
    assert empty_target_side.is_conclusive is False

    empty_theme_side = mod.compute_target_universe_disjointness(
        theme_keys=[], target_ids=["node:athena"]
    )
    assert empty_theme_side.is_conclusive is False

    both_empty = mod.compute_target_universe_disjointness([], [])
    assert both_empty.is_conclusive is False


def test_disjointness_both_sides_populated_is_conclusive() -> None:
    result = mod.compute_target_universe_disjointness(
        theme_keys=["open-loop-aaa"], target_ids=["node:athena"]
    )
    assert result.is_conclusive is True


# ===========================================================================
# compute_theme_first_seen
# ===========================================================================


def test_theme_first_seen_picks_earliest_per_theme() -> None:
    rows = [
        (_ts(10), "open-loop-aaa", 0.4),
        (_ts(8), "open-loop-aaa", 0.35),
        (_ts(12), "open-loop-bbb", 0.42),
    ]
    first_seen = mod.compute_theme_first_seen(rows)
    assert first_seen == {"open-loop-aaa": _ts(8), "open-loop-bbb": _ts(12)}


def test_theme_first_seen_skips_blank_theme_key() -> None:
    rows = [(_ts(10), "", 0.4), (_ts(11), None, 0.4)]  # type: ignore[list-item]
    assert mod.compute_theme_first_seen(rows) == {}


def test_theme_first_seen_empty_input() -> None:
    assert mod.compute_theme_first_seen([]) == {}


# ===========================================================================
# check_winner_traceability
# ===========================================================================


def test_winner_traceability_all_traceable() -> None:
    self_model_rows = [
        (_ts(10), "open-loop-aaa"),
        (_ts(11), None),
        (_ts(12), "open-loop-aaa"),
    ]
    first_seen = {"open-loop-aaa": _ts(9)}
    result = mod.check_winner_traceability(self_model_rows, first_seen)
    assert result.total_self_model_rows == 3
    assert result.rows_with_winner == 2
    assert result.traceable_count == 2
    assert result.untraceable_loop_ids == ()
    assert result.all_traceable is True


def test_winner_traceability_untraceable_when_id_unknown() -> None:
    self_model_rows = [(_ts(10), "open-loop-dangling")]
    result = mod.check_winner_traceability(self_model_rows, {})
    assert result.traceable_count == 0
    assert result.untraceable_loop_ids == ("open-loop-dangling",)
    assert result.all_traceable is False


def test_winner_traceability_untraceable_when_first_seen_is_after_tick() -> None:
    # AST/HOT names a winner at 10:00, but the trace table's earliest sighting
    # of that theme is 11:00 -- AST/HOT cannot honestly be "ahead of" its own
    # real source, so this must be reported untraceable, not assumed matched.
    self_model_rows = [(_ts(10), "open-loop-aaa")]
    first_seen = {"open-loop-aaa": _ts(11)}
    result = mod.check_winner_traceability(self_model_rows, first_seen)
    assert result.traceable_count == 0
    assert result.untraceable_loop_ids == ("open-loop-aaa",)


def test_winner_traceability_no_winners_present() -> None:
    self_model_rows = [(_ts(10), None), (_ts(11), None)]
    result = mod.check_winner_traceability(self_model_rows, {})
    assert result.rows_with_winner == 0
    assert result.traceable_count == 0
    assert result.all_traceable is False  # vacuous case must not read as "fully traceable"


def test_winner_traceability_naive_timezone_handled() -> None:
    # Real DB rows can come back tz-naive if a driver doesn't attach UTC --
    # the comparison must not raise or silently mis-order.
    naive_tick = datetime(2026, 7, 30, 10, 0)
    naive_first_seen = datetime(2026, 7, 30, 9, 0)
    result = mod.check_winner_traceability(
        [(naive_tick, "open-loop-aaa")], {"open-loop-aaa": naive_first_seen}
    )
    assert result.traceable_count == 1


# ===========================================================================
# resolve_live_salience_formula -- must use the SAME truthy set salience.py's
# own salience_v2_enabled() does (imported directly, not reimplemented).
# ===========================================================================


def test_resolve_live_formula_true_variants() -> None:
    for raw in ("true", "TRUE", "1", "yes", "on"):
        label, is_v2 = mod.resolve_live_salience_formula(raw)
        assert is_v2 is True
        assert label == mod.LIVE_FORMULA_V2


def test_resolve_live_formula_false_variants() -> None:
    for raw in ("false", "0", "no", "off", "garbage"):
        label, is_v2 = mod.resolve_live_salience_formula(raw)
        assert is_v2 is False
        assert label == mod.LIVE_FORMULA_FALLBACK


def test_resolve_live_formula_none_defaults_to_fallback() -> None:
    # Unset env var -- salience.py's own default is "false", must match here.
    label, is_v2 = mod.resolve_live_salience_formula(None)
    assert is_v2 is False
    assert label == mod.LIVE_FORMULA_FALLBACK


def test_resolve_live_formula_uses_real_truthy_set_from_salience_module() -> None:
    # Regression guard: this must import the real _TRUTHY set from
    # orion.substrate.attention.salience, not a hand-copied literal that
    # could silently drift from the production check.
    from orion.substrate.attention.salience import _TRUTHY as real_truthy

    assert mod.SALIENCE_TRUTHY_VALUES is real_truthy


# ===========================================================================
# write_traceability_csv -- pure-enough to check without mocking a real DB
# (writes to a tmp_path fixture, no network).
# ===========================================================================


def test_write_traceability_csv_marks_untraceable_rows(tmp_path) -> None:
    path = tmp_path / "out.csv"
    self_model_rows = [(_ts(10), "open-loop-aaa"), (_ts(9), "open-loop-dangling"), (_ts(8), None)]
    first_seen = {"open-loop-aaa": _ts(9)}
    mod.write_traceability_csv(path, self_model_rows, first_seen)
    content = path.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert lines[0] == "generated_at,broadcast_selected_open_loop_id,loop_first_seen_at,traceable"
    assert "open-loop-aaa" in lines[1]
    assert lines[1].endswith("True")
    assert "open-loop-dangling" in lines[2]
    assert lines[2].endswith("False")
