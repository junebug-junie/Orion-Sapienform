"""scripts/pr_field_note_cost.py: the measured per-note cost (spec acceptance check 6) summarizes only
completed runs, counts every attempt's tokens, and says plainly when nothing is measured yet."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pr_field_note_cost as cost  # noqa: E402


def row(i, terminal="completed", **kw):
    base = {"run_id": f"pr-field-note-{i}", "pr": i, "terminal": terminal, "turns": 1, "prompt": 50_000,
            "cached": 20_000, "completion": 3_000, "harness_steps": 20, "seconds": 200.0, "wh": 20.0,
            "gpu_slot_s": 200.0}
    return {**base, **kw}


def test_no_runs_says_the_table_is_still_an_estimate():
    assert "ESTIMATE" in "\n".join(cost.summarize([], 24))


def test_median_and_p90_over_completed_runs_only():
    rows = [row(i, seconds=100.0 * (i + 1)) for i in range(10)] + [row(99, "failed", seconds=9e9)]
    text = "\n".join(cost.summarize(rows, 24))
    assert "ended: 11 (10 completed)" in text
    line = next(ln for ln in text.splitlines() if "wall s" in ln)
    assert "median      550.0" in line and "p90      900.0" in line and "~13,200/day" in line
    assert "10 of the 20 notes" in text


def test_a_missing_join_reads_unread_not_zero():
    text = "\n".join(cost.summarize([row(1, wh=None)], 24))
    assert "Wh, whole machine above rest" in text and "UNREAD" in text


def test_the_sql_counts_every_attempts_turn_and_whole_machine_energy():
    assert "step_correlation_id" in cost.SQL and "wh_by_host" in cost.SQL and "durable.pr_field_note" in cost.SQL
