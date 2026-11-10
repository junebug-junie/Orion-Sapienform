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
    assert "10 of the 20 notes the design asks for" in text


def test_a_missing_join_reads_unread_not_zero():
    text = "\n".join(cost.summarize([row(1, wh=None)], 24))
    assert "Wh, whole machine above rest" in text and "UNREAD" in text


def test_the_sql_counts_every_attempts_turn_and_whole_machine_energy():
    assert "step_correlation_id" in cost.SQL and "wh_by_host" in cost.SQL and "durable.pr_field_note" in cost.SQL


def test_a_null_energy_map_reads_unread_and_does_not_crash_the_report():
    # live 2026-11-06: one act had "wh_by_host": null; summing it in SQL failed the whole query
    # ("cannot call jsonb_each_text on a non-object"), so acceptance check 6 could not run at all
    r = cost.with_sums({"run_id": "pr-field-note-1", "terminal": "completed", "wh_by_host": None,
                        "gpu_slot_s_by_lane": {"agent-gpu2": 693.793}})
    assert r["wh"] is None and r["gpu_slot_s"] == 693.793
    assert "wh_by_host" not in r and "gpu_slot_s_by_lane" not in r
    assert cost.map_sum({"circe": 3.25, "athena": 1.0}) == 4.25
    assert cost.map_sum({}) is None and cost.map_sum([1, 2]) is None and cost.map_sum("3") is None


def test_the_sql_no_longer_sums_maps_that_may_be_json_null():
    assert "jsonb_each_text" not in cost.SQL


def test_a_partly_unreadable_map_reads_unread_not_a_partial_sum():
    assert cost.map_sum({"circe": None, "athena": 1.0}) is None
    assert cost.map_sum({"circe": "2.5"}) == 2.5 and cost.map_sum({"circe": "n/a"}) is None
    assert cost.map_sum({"circe": True}) is None


def test_the_reminiscence_query_counts_its_own_runs_and_acts():
    sql = cost.sql_for("reminiscence.episode", 30)
    assert "'reminiscence.episode'" in sql and "'durable.reminiscence'" in sql and "make_interval(days => 30)" in sql
    assert "'pr.field_note'" not in sql and "durable.pr_field_note" not in sql
    assert "step_correlation_id" in sql   # every step turn of the run (six a day), every attempt


def test_the_reminiscence_summary_projects_one_episode_a_day_and_asks_for_seven():
    rows = [row(i, seconds=1800.0, gpu_slot_s=1800.0) for i in range(3)]
    text = "\n".join(cost.summarize(rows, 1.0, "reminiscence.episode"))
    assert text.startswith("reminiscence.episode runs ended: 3 (3 completed)")
    assert "~1,800/day at 1 episodes" in next(ln for ln in text.splitlines() if "GPU slot s" in ln)
    assert "3 of the 7 episodes the design asks for" in text
