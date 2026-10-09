"""Read-only focus exposure report, or legacy winner-sequence recorder replay.

Completed rows: history.jsonl --runs --start ISO --end ISO.
Use --print-sql --start ISO --end ISO to print the bounded read-only export.

Usage: python services/orion-attention-runtime/evals/replay_focus_runs.py history.jsonl
Input: JSONL export with observed_at, target_id, streak_count, min_streak_at_tick,
source_field_tick_id and source_attention_frame_id. No database connection or writes.
The oracle independently groups adjacent targets. Completed rows must match exactly;
no-winner spans and the still-open tail must never masquerade as completed focus.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import groupby
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "services/orion-attention-runtime"), str(REPO)]
from statistics import median


# Exact producer ids, not a substring taxonomy. transport is a retired bus sensor.
INTERNAL_TARGETS = {"node:substrate.biometrics", "node:substrate.bus_synaptic",
                    "node:substrate.transport"}


def utc(value):
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def exposure(rows, start, end):
    """Completed goal-provenance spans, NOT overall attention or continuous activity.

    A wall span may include downtime. Never infer no-winner ticks from a gap or
    prorate tick counts across a boundary. Censored spans contribute exposure,
    but not the complete-duration distribution.
    """
    start, end = utc(start), utc(end)
    if end <= start:
        raise ValueError("end must follow start")
    runs, seen = [], set()
    for row in rows:
        if row["run_id"] in seen:
            raise ValueError("duplicate run_id")
        seen.add(row["run_id"])
        a, b = utc(row["started_at"]), utc(row["ended_at"])
        if b < a or row["tick_count"] < 1 or not isinstance(row["left_censored"], bool):
            raise ValueError("invalid completed run")
        if a < end and b > start:
            runs.append((max(a, start), min(b, end), row, a, b))
    runs.sort(key=lambda r: r[0])
    if any(a[1] > b[0] for a, b in zip(runs, runs[1:])):
        raise ValueError("overlapping runs: cannot sum attention exposure")
    targets = {}
    for a, b, row, original_a, original_b in runs:
        item = targets.setdefault(row["target_id"], dict(
            runs=0, wall_span_seconds=0, full_window_ticks=0, complete_durations_seconds=[]))
        item["runs"] += 1
        item["wall_span_seconds"] += (b - a).total_seconds()
        if original_a >= start and original_b <= end:
            item["full_window_ticks"] += row["tick_count"]
            if not row["left_censored"]:
                item["complete_durations_seconds"].append((b - a).total_seconds())
    total_span = sum(v["wall_span_seconds"] for v in targets.values())
    ticks = sum(v["full_window_ticks"] for v in targets.values())
    for item in targets.values():
        durations = item.pop("complete_durations_seconds")
        item.update(complete_duration_count=len(durations),
                    median_complete_seconds=median(durations) if durations else None,
                    longest_complete_seconds=max(durations) if durations else None,
                    share_of_recorded_span=item["wall_span_seconds"] / total_span if total_span else None,
                    share_of_window=item["wall_span_seconds"] / (end - start).total_seconds())
    days, rolling = [], []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        lo, hi = max(day, start), min(day + timedelta(days=1), end)
        spans = Counter()
        for a, b, row, _, _ in runs:
            spans[row["target_id"]] += max(0, (min(b, hi) - max(a, lo)).total_seconds())
        days.append(dict(day=day.date().isoformat(), window_seconds=(hi-lo).total_seconds(),
                         wall_span_seconds=dict(spans),
                         share_of_utc_day={k: v / 86400 for k, v in spans.items()},
                         unaccounted_seconds=(hi-lo).total_seconds()-sum(spans.values())))
        day += timedelta(days=1)
    # Trailing 30-minute windows, stepped every five minutes, full windows only.
    at = start + timedelta(minutes=30)
    while at <= end:
        spans = Counter()
        for a, b, row, _, _ in runs:
            seconds = max(0, (min(b, at)-max(a, at-timedelta(minutes=30))).total_seconds())
            if seconds:
                spans[row["target_id"]] += seconds
        rolling.append(dict(ended_at=at.isoformat(),
                            share_of_window={k: v / 1800 for k, v in spans.items()},
                            unaccounted_seconds=1800-sum(spans.values())))
        at += timedelta(minutes=5)
    other_runs = sum(r[2]["target_id"] not in INTERNAL_TARGETS for r in runs)
    other_ticks = sum(v["full_window_ticks"] for k, v in targets.items() if k not in INTERNAL_TARGETS)
    return dict(start=start.isoformat(), end=end.isoformat(),
                scope="goal-provenance node winner; not overall attention",
                caveats=["Wall spans can include downtime; they are not continuous activity.",
                         "Open tail, no-winner and recorder-off periods are unaccounted, not idle.",
                         "Tick share excludes runs clipped by the window; duration excludes left censoring."],
                completed_runs=len(runs), targets=targets, days=days, rolling_30_minutes=rolling,
                other_winner_runs=other_runs, other_run_share=other_runs/len(runs) if runs else None,
                full_window_winner_ticks=ticks, other_winner_ticks=other_ticks,
                other_tick_share=other_ticks/ticks if ticks else None,
                unaccounted_seconds=(end-start).total_seconds()-total_span,
                verdict="UNVERIFIED: habituation needs seven post-change days and overall-winner coverage")


def export_sql(start, end):
    start, end = utc(start), utc(end)
    if end <= start:
        raise ValueError("end must follow start")
    return f"""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL TIME ZONE 'UTC';
SELECT row_to_json(r) FROM (
 SELECT * FROM field_dominance_run
 WHERE started_at < '{end.isoformat()}'::timestamptz
   AND ended_at > '{start.isoformat()}'::timestamptz ORDER BY started_at, run_id
) r;
ROLLBACK;"""


def replay(rows: list[dict]) -> dict:
    from app.dominance_runs import advance_run

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, nargs="?")
    parser.add_argument("--runs", action="store_true", help="report completed focus rows instead of legacy ticks")
    parser.add_argument("--start", type=utc)
    parser.add_argument("--end", type=utc)
    parser.add_argument("--print-sql", action="store_true", help="print a read-only JSONL export query")
    args = parser.parse_args()
    if (args.runs or args.print_sql) and not (args.start and args.end):
        parser.error("--start and --end are required for a bounded report")
    if args.print_sql:
        print(export_sql(args.start, args.end))
        return
    if not args.history:
        parser.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    print(json.dumps(exposure(rows, args.start, args.end) if args.runs else replay(rows), indent=2))


if __name__ == "__main__":
    main()
