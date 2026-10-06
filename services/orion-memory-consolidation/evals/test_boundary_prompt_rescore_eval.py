"""Offline pieces of run_boundary_prompt_rescore_eval.py: the before/after prompts differ only by
the BOUNDARY definition, and Rule 3 replay splits as boundary.py does."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_boundary_prompt_rescore_eval as ev  # noqa: E402
from orion.memory.turn_change_classify import BOUNDARY_DEFINITION  # noqa: E402


def test_before_is_after_without_the_definition():
    prev = {"prompt": "p" * 400, "response": "r"}
    turn = {"prompt": "next", "response": "ok", "temporal_phase": None}
    before, after = ev.prompts_for(turn, prev)
    assert BOUNDARY_DEFINITION not in before and BOUNDARY_DEFINITION in after
    assert before == after.replace(BOUNDARY_DEFINITION, "")
    assert "phase=unknown" in after and ("p" * 297 + "...") in after  # live clip at 300


def test_rule3_replay_matches_boundary_rules():
    turns = ev.annotate([
        {"at": "2026-10-05T02:55:00+00:00", "before": None, "after": None},
        {"at": "2026-10-05T03:00:00+00:00", "before": 0.99, "after": 0.99},   # short_pause: never
        {"at": "2026-10-05T03:40:00+00:00", "before": 0.99, "after": 0.10},   # resumed_thread: score decides
    ])
    assert [t["phase"] for t in turns][1:] == ["short_pause", "resumed_thread"]
    assert len(ev.split(turns, "before")) == 2
    assert len(ev.split(turns, "after")) == 1
    r = ev.report(turns)
    assert r["before"]["resumed_thread_ge_0_92"] == 1 and r["after"]["resumed_thread_ge_0_92"] == 0
    assert r["after"]["sessions"]["chicago_2026_10_05"]["episodes"] == 1


def _t(at, **kw):
    return {"at": at, **kw}


def test_live_legacy_rule_closes_on_llm_or_time_gap_and_seeds_next_window():
    turns = ev.annotate([
        _t("2026-10-05T02:00:00+00:00", before=None),
        _t("2026-10-05T02:05:00+00:00", before=0.90),   # >= 0.85: closes, seeds next window
        _t("2026-10-05T02:10:00+00:00", before=0.10),
        _t("2026-10-05T04:00:00+00:00", before=0.10),   # 110 min gap >= 5400 s: time_gap
    ])
    s = ev.legacy_summary(turns, "before")
    assert s["closed_by"] == {"llm": 1, "time_gap": 1}
    assert s["windows"] == 3


def test_consumer_flags_follow_live_thresholds():
    f = ev.consumer_flags({"novelty_score": 0.7, "confidence": 0.4, "shift_kind": "STANCE",
                           "memory_significance_score": 0.5, "conversation_boundary_score": 0.9})
    assert f["turn_change_signal_emitted"] and f["substantive_shift"] and f["retrieval_intent_relational"]
    assert f["live_window_close_ge_0_85"] and f["memory_significance_ge_0_40"]
    assert not f["recall_novelty_below_floor"] and not f["retrieval_intent_open_loop"]
    g = ev.consumer_flags({"novelty_score": 0.1, "shift_kind": "NONE"})
    assert g["recall_novelty_below_floor"] and not g["substantive_shift"]


def test_other_lines_report_counts_flips_against_noise():
    a = {"novelty_score": 0.9, "confidence": 0.8, "shift_kind": "TOPIC", "memory_significance_score": 0.9,
         "conversation_boundary_score": 0.9}
    b = {**a, "shift_kind": "NONE", "conversation_boundary_score": 0.0}
    turns = [{"scores": {"before_r0": a, "before_r1": a, "after_r0": b, "after_r1": b}}]
    r = ev.other_lines_report(turns, 2)
    assert r["per_turn_flips"]["noise_before_r0_vs_r1"]["shift_kind"] == 0
    assert r["per_turn_flips"]["before_vs_after"]["substantive_shift"] == 1
