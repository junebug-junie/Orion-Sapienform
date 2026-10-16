"""Reduce + summarize logic of measure_personality_choices, on in-memory rows (no DB)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_personality_choices import (  # noqa: E402
    LANE_CURIOSITY,
    LANE_OVERRIDE,
    reduce_rows,
    render_summary,
    summarize,
)


def _curiosity_row(run_id, tested, *, joined=True):
    offered = [{"rank": i, "prior_id": f"p{i}", "expected_nats": 0.1} for i in range(1, 9)]
    row = {
        "run_id": run_id, "decided_at": "2026-10-16T10:00:00+00:00", "arm": "value_order",
        "offered": offered, "stale_offered": [{"rank": 1, "prior_id": "s1"}],
        "material_ids": [], "turn_started_at": "2026-10-16T11:00:00+00:00",
        "outcome_run_id": None,
    }
    if joined:
        row.update(
            outcome_run_id=run_id, completed_at="2026-10-16T11:10:00+00:00", turn_ok=True,
            realized_nats=0.2, per_prior=[{"kind": "tested", "prior_id": p} for p in tested],
        )
    return row


def _override_row(log_id, vo):
    return {
        "log_id": log_id, "generated_at": "2026-10-16T12:00:00+00:00",
        "projection_json": {"frame": {"voluntary_override": vo, "open_loops": [
            {"id": "a", "salience": 0.9, "source_refs": ["node:substrate.perception"]},
            {"id": "b", "salience": 0.8, "source_refs": ["node:substrate.execution"]},
        ]}},
    }


VO = {"chosen_loop_id": "b", "beat_loop_id": "a", "effort_spent": 0.5,
      "goal_artifact_id": "artifact-1"}


def test_reduce_and_summarize_buckets_and_ranks():
    cur = [
        _curiosity_row("r1", ["p1"]),              # followed the order
        _curiosity_row("r2", ["p3", "s1"]),        # deviated; one stale-section pick
        _curiosity_row("r3", []),                  # nothing tested
        _curiosity_row("r4", ["p1"], joined=False),  # no outcome yet
    ]
    ovr = [_override_row("L1", VO), _override_row("L2", {"chosen_loop_id": "b"})]
    choices, skipped = reduce_rows(cur, ovr)
    report = summarize(choices, skipped, window_days=14,
                       rows_read={LANE_CURIOSITY: 4, LANE_OVERRIDE: 2, "broadcast_rows_total": 9})
    c = report["lanes"][LANE_CURIOSITY]
    assert c["choices"] == 2
    assert c["skipped"] == {"no_outcome_row": 1, "nothing_tested": 1}
    assert c["chosen_best_rank"] == {"1": 1, "3": 1}
    assert c["followed_order_rank1"] == 1
    assert c["picks_total"] == 3 and c["picks_outside_ordered_list"] == 1
    assert c["mean_alternatives_per_choice"] == 7.5  # (9-1 + 9-2) / 2
    assert c["mean_cost_seconds"] == 600.0
    o = report["lanes"][LANE_OVERRIDE]
    assert o["choices"] == 1 and o["skipped"] == {"malformed_override": 1}
    assert o["chosen_best_rank"] == {"2": 1} and o["followed_order_rank1"] == 0
    assert o["mean_cost_effort"] == 0.5
    text = render_summary(report)
    assert "took rank 1: 1/2 (50%)" in text
    assert "no_outcome_row=1, nothing_tested=1" in text
    assert "override or not: 9" in text


def test_summarize_empty():
    report = summarize([], {LANE_CURIOSITY: {}, LANE_OVERRIDE: {}}, window_days=14, rows_read={})
    assert report["lanes"][LANE_CURIOSITY]["mean_alternatives_per_choice"] is None
    assert "yielded None: 0" in render_summary(report)
