"""Read-only focus exposure report, or legacy winner-sequence recorder replay.

Completed rows: history.jsonl --runs --start ISO --end ISO.
R1a hogging check (PR #2369 rev 4): history.jsonl --r1a --start ISO --end ISO.
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


def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def hogging_stretches(rolling, *, share=0.5):
    """Consecutive rolling windows in which one target held MORE than `share`.

    `seconds` is how long the 30-minute share series stayed above the bar:
    last qualifying window end - first qualifying window end + one 5-minute step.
    The covered wall span is reported separately. Missing/unaccounted seconds count against the share (never as the
    target's), so a recorder outage can only shorten a stretch, never make one.
    """
    out, open_ = [], {}
    for i, w in enumerate(rolling):
        end = utc(w["ended_at"])
        hogs = {t for t, v in w["share_of_window"].items() if v > share}
        contiguous = i > 0 and (end - utc(rolling[i - 1]["ended_at"])) <= timedelta(minutes=5)
        for target in list(open_):
            if target not in hogs or not contiguous:
                out.append(open_.pop(target))
        for target in hogs:
            peak = w["share_of_window"][target]
            if target in open_:
                item = open_[target]
                item.update(last_window_end=end, windows=item["windows"] + 1,
                            peak_share=max(item["peak_share"], peak))
            else:
                open_[target] = dict(target_id=target, first_window_start=end - timedelta(minutes=30),
                                     last_window_end=end, windows=1, peak_share=peak)
    out.extend(open_.values())
    for item in out:
        first_end = item["first_window_start"] + timedelta(minutes=30)
        item["seconds"] = (item["last_window_end"] - first_end).total_seconds() + 300
        item["covered_wall_seconds"] = (item["last_window_end"] - item["first_window_start"]).total_seconds()
        item["first_window_start"] = item["first_window_start"].isoformat()
        item["last_window_end"] = item["last_window_end"].isoformat()
    return sorted(out, key=lambda r: -r["seconds"])


def merged_arcs(rows, start, end, return_minutes):
    """Patch-1 arc rule probe: same-target runs whose gap is <= R merge as returns."""
    start, end = utc(start), utc(end)
    runs = sorted((r for r in rows if utc(r["ended_at"]) > start and utc(r["started_at"]) < end),
                  key=lambda r: utc(r["started_at"]))
    last_end, arcs = {}, Counter()
    returns = Counter()
    for r in runs:
        t, a = r["target_id"], utc(r["started_at"])
        if t in last_end and (a - last_end[t]).total_seconds() <= return_minutes * 60:
            returns[t] += 1
        else:
            arcs[t] += 1
        last_end[t] = max(last_end.get(t, a), utc(r["ended_at"]))
    total_arcs = sum(arcs.values())
    return dict(return_minutes=return_minutes, arcs=total_arcs, arcs_by_target=dict(arcs),
                returns_by_target=dict(returns),
                mean_returns_per_arc=sum(returns.values()) / total_arcs if total_arcs else None)


def r1a_report(rows, start, end, *, share=0.5, hog_hours=2.0, min_streak=3,
               stuck_hours=4.0, return_minutes=(1, 2, 5, 10, 30), instrument_since=None,
               min_coverage=0.8):
    """R1a: does any target hold > `share` of 30-minute windows for > `hog_hours`?

    Also the patch-1 interoception-lane distributions computed from the same
    rows: run lengths, share of runs reaching the arc's minimum streak, runs
    longer than `stuck_hours`, and arc counts under candidate return windows R.
    `instrument_since` marks the first instant of the instrument R1a is meant to
    judge (#2528 live); without it, or with too little history after it, the
    verdict is a baseline only and never a build/no-build decision.
    """
    base = exposure(rows, start, end)
    start, end = utc(start), utc(end)
    inside = [r for r in rows if utc(r["started_at"]) >= start and utc(r["ended_at"]) <= end]
    ticks = sorted(r["tick_count"] for r in inside)
    complete = sorted((utc(r["ended_at"]) - utc(r["started_at"])).total_seconds()
                      for r in inside if not r["left_censored"])
    stretches = hogging_stretches(base["rolling_30_minutes"], share=share)
    longest = {}
    for s in stretches:
        longest[s["target_id"]] = max(longest.get(s["target_id"], 0), s["seconds"])
    hogs = [s for s in stretches if s["seconds"] > hog_hours * 3600]
    recorded = (end - start).total_seconds() - base["unaccounted_seconds"]
    judged_from = utc(instrument_since) if instrument_since else None
    days_on_instrument = ((end - max(start, judged_from)).total_seconds() / 86400
                          if judged_from and judged_from < end else 0.0)
    # The decision only looks at the judged instrument: hogs before it never count.
    judged_hogs, judged_coverage = [], None
    if judged_from and judged_from < end:
        judged = exposure(rows, max(start, judged_from), end)
        span = (end - max(start, judged_from)).total_seconds()
        judged_coverage = (span - judged["unaccounted_seconds"]) / span
        judged_hogs = [s for s in hogging_stretches(judged["rolling_30_minutes"], share=share)
                       if s["seconds"] > hog_hours * 3600]
    if base["completed_runs"] == 0:
        verdict = "NO_DATA: no completed focus runs in the window"
    elif days_on_instrument < 7:
        verdict = ("BASELINE_ONLY: R1a judges the post-#2528 instrument after 7 live days; "
                   f"{days_on_instrument:.2f} days available. "
                   + ("A pre-#2528 target exceeded the hog bar." if hogs else
                      "No target exceeded the hog bar in this baseline."))
    elif judged_hogs:
        verdict = "BUILD_R1: a target hogged attention on the judged instrument"
    elif judged_coverage < min_coverage:
        verdict = (f"INSUFFICIENT_COVERAGE: completed runs cover {judged_coverage:.1%} of the judged "
                   f"window (< {min_coverage:.0%}); missing time is not calm")
    else:
        verdict = "DO_NOT_BUILD_R1: no target hogged attention (open tail at export is UNVERIFIED)"
    return dict(start=start.isoformat(), end=end.isoformat(), completed_runs=base["completed_runs"],
                recorded_hours=recorded / 3600, unaccounted_hours=base["unaccounted_seconds"] / 3600,
                instrument_since=judged_from.isoformat() if judged_from else None,
                days_on_instrument=days_on_instrument, judged_coverage=judged_coverage,
                judged_hog_stretches=judged_hogs,
                hog_rule=dict(share_above=share, longer_than_hours=hog_hours, window_minutes=30, step_minutes=5),
                hog_stretches=hogs, longest_majority_stretch_seconds_by_target=longest,
                share_of_recorded_span={k: v["share_of_recorded_span"] for k, v in base["targets"].items()},
                run_ticks=dict(n=len(ticks), p50=_percentile(ticks, .5), p90=_percentile(ticks, .9),
                               p99=_percentile(ticks, .99), max=ticks[-1] if ticks else None),
                complete_run_seconds=dict(n=len(complete), p50=_percentile(complete, .5),
                                          p90=_percentile(complete, .9), max=complete[-1] if complete else None),
                share_runs_reaching_min_streak=(sum(t >= min_streak for t in ticks) / len(ticks)) if ticks else None,
                min_streak_assumed=min_streak,
                stuck_runs=[dict(target_id=r["target_id"], started_at=r["started_at"], ended_at=r["ended_at"],
                                 tick_count=r["tick_count"]) for r in inside
                            if (utc(r["ended_at"]) - utc(r["started_at"])).total_seconds() > stuck_hours * 3600],
                return_window_probe=[merged_arcs(rows, start, end, m) for m in return_minutes],
                verdict=verdict,
                caveats=base["caveats"] + [
                    "Share denominator is the full 30-minute window; unaccounted time never counts toward a hog.",
                    "Runs before and after a scoring change are different instruments; pass instrument_since.",
                    "Only completed runs are stored: a run still open at export (a live hog) is invisible.",
                    "Percentiles use the upper index (p50 of [1, 2] is 2)."])


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
    parser.add_argument("--r1a", action="store_true", help="R1a hogging verdict plus run-length distributions")
    parser.add_argument("--instrument-since", type=utc, help="first instant of the instrument R1a judges (#2528 live)")
    parser.add_argument("--print-sql", action="store_true", help="print a read-only JSONL export query")
    args = parser.parse_args()
    if (args.runs or args.r1a or args.print_sql) and not (args.start and args.end):
        parser.error("--start and --end are required for a bounded report")
    if args.print_sql:
        print(export_sql(args.start, args.end))
        return
    if not args.history:
        parser.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    if args.r1a:
        result = r1a_report(rows, args.start, args.end, instrument_since=args.instrument_since)
    else:
        result = exposure(rows, args.start, args.end) if args.runs else replay(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
