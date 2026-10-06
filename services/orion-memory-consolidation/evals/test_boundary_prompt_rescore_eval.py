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
