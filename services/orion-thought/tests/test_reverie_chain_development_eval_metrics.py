"""Pure metrics of evals/reverie_chain_development_eval.py (no database)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1] / "evals" / "reverie_chain_development_eval.py"


def _mod():
    spec = importlib.util.spec_from_file_location("reverie_chain_development_eval", _EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bigram_overlap_bounds():
    m = _mod()
    assert m.bigram_overlap("a b c", "a b c") == 1.0
    assert m.bigram_overlap("a b c", "x y z") == 0.0
    assert m.bigram_overlap("a", "a b") is None


def test_summarize_counts_and_baseline():
    m = _mod()
    rows = [
        ("c1", 0, "ol-1", "The coalition is a b c d", "next", "2026-10-16T00:00"),
        ("c1", 1, "ol-1", "The coalition is a b c d", "", "2026-10-16T00:01"),
        ("c2", 0, "ol-1", "something else unattended for days", "", "2026-10-16T01:00"),
    ]
    r = m.summarize(rows)
    assert r["chains"] == 2
    assert r["steps_per_chain"] == {1: 1, 2: 1}
    assert r["median_bigram_overlap_first_vs_last_step"] == 1.0
    assert r["cross_chain_pairs"] == 1
    assert r["same_subject_cross_chain_baseline"] == 0.0
    assert r["pct_opening_the_coalition"] == round(200 / 3, 1)
    assert r["next_focus_asked_steps"] == 1
    assert r["next_focus_fill_rate_pct"] == 100.0
    assert r["chains_last_step_without_next_focus"] == 2
    assert r["thoughts_saying_unattended_for"] == 1
