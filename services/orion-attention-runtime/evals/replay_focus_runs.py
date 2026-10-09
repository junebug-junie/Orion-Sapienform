"""Read-only replay of an exported legacy winner sequence through the new recorder.

Usage: python services/orion-attention-runtime/evals/replay_focus_runs.py history.jsonl
Input: JSONL export with observed_at, target_id, streak_count, min_streak_at_tick,
source_field_tick_id and source_attention_frame_id. No database connection or writes.
The oracle independently groups adjacent targets. Completed rows must match exactly;
no-winner spans and the still-open tail must never masquerade as completed focus.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from itertools import groupby
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "services/orion-attention-runtime"), str(REPO)]
from app.dominance_runs import advance_run


def replay(rows: list[dict]) -> dict:
    groups = [list(g) for _, g in groupby(rows, key=lambda r: r["target_id"])]
    expected = []
    for i, group in enumerate(groups[:-1]):
        first, last = group[0], group[-1]
        if first["target_id"] is not None:
            expected.append((first["target_id"], len(group), first["source_attention_frame_id"],
                last["source_attention_frame_id"], datetime.fromisoformat(first["observed_at"]),
                datetime.fromisoformat(groups[i + 1][0]["observed_at"])))
    state = None
    actual = []
    for i, row in enumerate(rows):
        if i % 997 == 0 and state:
            state = json.loads(json.dumps(state))  # simulate a process restart
        state, closed = advance_run(state, target_id=row["target_id"],
            target_kind="node" if row["target_id"] else None,
            observed_at=datetime.fromisoformat(row["observed_at"]),
            field_tick_id=row["source_field_tick_id"], frame_id=row["source_attention_frame_id"],
            min_streak=row["min_streak_at_tick"],
            left_censored=i == 0 and row["streak_count"] > 1)
        if closed:
            actual.append((closed.target_id, closed.tick_count, closed.first_source_attention_frame_id,
                closed.last_source_attention_frame_id, closed.started_at, closed.ended_at))
    assert actual == expected, "completed runs differ from independent winner grouping"
    lengths = sorted(r[1] for r in actual)
    return dict(input_ticks=len(rows), completed_runs=len(actual), exact_match=True,
                no_winner_ticks=sum(r["target_id"] is None for r in rows),
                runs_by_target=dict(Counter(r[0] for r in actual)),
                min_ticks=min(lengths, default=0), max_ticks=max(lengths, default=0),
                median_ticks=lengths[len(lengths) // 2] if lengths else 0,
                open_tail_excluded=bool(state and state["active"]),
                first_run_left_censored=bool(rows and rows[0]["streak_count"] > 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    args = parser.parse_args()
    with args.history.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(json.dumps(replay(rows), indent=2))
