#!/usr/bin/env python3
"""Read saved dream pressure alongside actual cycles; never run/recompute a dream.

JSONL input is the minimal projection printed by --print-sql. The store saves
pressure only on cycles in legacy exports. --with-checks includes the new real
check observations; missing checks never establish reset or recovery.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median


def utc(value):
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def export_sql(start, end, *, with_checks=False):
    start, end = utc(start), utc(end)
    if end <= start:
        raise ValueError("end must follow start")
    checks_sql = f"""
SELECT jsonb_build_object('kind', 'pressure_check', 'observation', observation_json)
FROM dream_pressure_observation
WHERE observed_at >= '{start.isoformat()}'::timestamptz
  AND observed_at < '{end.isoformat()}'::timestamptz ORDER BY observed_at;
""" if with_checks else ""
    return f"""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL TIME ZONE 'UTC';
SELECT row_to_json(r) FROM (
 SELECT cycle_id, started_at, ended_at, trigger, status,
        cycle_json->'pressure' AS reading
 FROM dream_cycle
 WHERE started_at < '{end.isoformat()}'::timestamptz
   AND (started_at >= '{start.isoformat()}'::timestamptz OR cycle_id = (
     SELECT cycle_id FROM dream_cycle WHERE started_at < '{start.isoformat()}'::timestamptz
     ORDER BY started_at DESC LIMIT 1))
 ORDER BY started_at, cycle_id
) r;
{checks_sql}
ROLLBACK;"""


def check_history(checks, cycles, start, end):
    """Use only real observations. Failed source reads break continuity."""
    checks = sorted((c for c in checks if start <= utc(c["observed_at"]) < end),
                    key=lambda c: utc(c["observed_at"]))
    if len({c["check_id"] for c in checks}) != len(checks):
        raise ValueError("duplicate pressure check")
    valid, invalid, gaps, zero, eligible_low = 0, 0, 0, 0, 0
    examples, discharged_windows = [], set()
    previous = None
    for c in checks:
        p = c["reading"]
        if c["source_errors"]:
            invalid += 1
            previous = None
            discharged_windows.clear()
            continue
        value = float(p["pressure"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid check pressure")
        valid += 1
        zero += value == 0
        last_end = utc(c["last_attempt_end"]) if c["last_attempt_end"] else None
        timer_clear = last_end is None or (utc(c["observed_at"])-last_end).total_seconds() >= c["min_interval_hours"]*3600
        eligible_low += (not c["forced"] and timer_clear and p["idle_minutes"] is not None
                         and p["idle_minutes"] >= p["idle_required_minutes"] and value < p["threshold"])
        window = c["last_window_start"]
        if previous:
            gap = (utc(c["observed_at"])-utc(previous["observed_at"])).total_seconds()
            continuous = gap <= 2*max(c["check_interval_sec"], previous["check_interval_sec"])
            gaps += not continuous
            same_formula = c["formula"] == previous["formula"]
            if continuous and same_formula:
                old = previous["reading"]["pressure"]
                successful_sleep = any(
                    r["status"] in {"completed", "empty"} and window is not None
                    and utc(r["started_at"]) == utc(window) and r.get("ended_at")
                    and utc(previous["observed_at"]) <= utc(r["started_at"])
                    and utc(r["ended_at"]) <= utc(c["observed_at"]) for r in cycles)
                if value < old and successful_sleep:
                    discharged_windows.add(utc(window))
                    examples.append(dict(kind="fall_after_cycle", at=c["observed_at"], before=old, after=value))
                elif (value > old and window is not None and utc(window) in discharged_windows
                      and previous["last_window_start"] is not None
                      and utc(previous["last_window_start"]) == utc(window)):
                    examples.append(dict(kind="rise_after_observed_fall", at=c["observed_at"], before=old, after=value))
            else:
                discharged_windows.clear()
        previous = c
    return dict(samples=len(checks), valid_samples=valid, source_failed_samples=invalid,
                cadence_gaps=gaps, zero_samples=zero, idle_timer_clear_below_threshold=eligible_low,
                observed_fall_and_rise=any(e["kind"] == "rise_after_observed_fall" for e in examples),
                examples=examples[:20])


def report(rows, start, end, *, interval_hours=6.0, check_seconds=600.0, legacy_cap=50):
    start, end = utc(start), utc(end)
    if end <= start or not math.isfinite(interval_hours) or interval_hours <= 0:
        raise ValueError("invalid window or minimum interval")
    if not math.isfinite(check_seconds) or check_seconds <= 0 or legacy_cap <= 0:
        raise ValueError("invalid cadence or legacy cap")
    checks = [r["observation"] for r in rows if r.get("kind") == "pressure_check"]
    rows = sorted((r for r in rows if r.get("kind") != "pressure_check"), key=lambda r: utc(r["started_at"]))
    if len({r["cycle_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate cycle_id")
    selected, previous_end = [], None
    for row in rows:
        at = utc(row["started_at"])
        ended = utc(row["ended_at"]) if row.get("ended_at") else None
        if ended is not None and ended < at:
            raise ValueError("cycle ends before it starts")
        if at >= end:
            break
        if at >= start:
            p = row["reading"]
            value, threshold = float(p["pressure"]), float(p["threshold"])
            if not all(math.isfinite(v) and v >= 0 for v in (value, threshold)):
                raise ValueError("invalid stored pressure")
            novelty = "new_counts" in p
            gap = (at-previous_end).total_seconds() if previous_end is not None else None
            delay = gap - interval_hours*3600 if gap is not None else None
            counts = p.get("counts", {})
            # The old four loaders each capped at 50; compaction weighs 0.5,
            # the other sources at most 1. Do not apply this ceiling to novelty.
            ceiling = legacy_cap * 3.5 if not novelty else None
            selected.append(dict(cycle_id=row["cycle_id"], started_at=at.isoformat(),
                ended_at=ended.isoformat() if ended else None, status=row["status"], trigger=row["trigger"],
                formula="novelty" if novelty else "legacy",
                pressure=value, threshold=threshold, threshold_ratio=value/threshold if threshold else None,
                counts=counts, new_counts=p.get("new_counts"), idle_minutes=p.get("idle_minutes"),
                hours_since_previous_attempt_end=gap/3600 if gap is not None else None,
                seconds_after_timer=delay,
                timer_aligned=(0 <= delay <= check_seconds) if delay is not None else None,
                legacy_ceiling=ceiling, fraction_of_legacy_ceiling=value/ceiling if ceiling else None,
                legacy_sources_at_cap=[k for k, n in counts.items() if n >= legacy_cap] if not novelty else None))
        if ended is not None:
            previous_end = max(previous_end, ended) if previous_end else ended
    versions = {}
    for version in ("legacy", "novelty"):
        values = [r for r in selected if r["formula"] == version]
        automatic = [r for r in values if r["trigger"] == "pressure" and r["seconds_after_timer"] is not None]
        pressures = [r["pressure"] for r in values]
        gaps = [r["hours_since_previous_attempt_end"] for r in automatic]
        versions[version] = dict(cycles=len(values), statuses=dict(Counter(r["status"] for r in values)),
            min_pressure=min(pressures) if pressures else None, max_pressure=max(pressures) if pressures else None,
            below_threshold_at_cycle=sum(r["pressure"] < r["threshold"] for r in values),
            comparable_automatic_cycles=len(automatic),
            comparable_automatic_below_threshold=sum(r["pressure"] < r["threshold"] for r in automatic),
            timer_aligned_cycles=sum(r["timer_aligned"] for r in automatic),
            median_gap_hours=median(gaps) if gaps else None)
    legacy = versions["legacy"]
    consistent = (legacy["comparable_automatic_cycles"] > 0
                  and legacy["timer_aligned_cycles"] == legacy["comparable_automatic_cycles"]
                  and legacy["comparable_automatic_below_threshold"] == 0)
    curve = check_history(checks, rows, start, end)
    return dict(start=start.isoformat(), end=end.isoformat(), cycles=selected, by_formula=versions,
        interval_hours_assumed=interval_hours, check_seconds_assumed=check_seconds,
        legacy_candidate_cap_assumed=legacy_cap,
        legacy_timing_consistent_with_timer=consistent,
        independent_check_samples=curve["samples"], check_history=curve,
        verdict=("Observed pressure fall after a completed cycle and a later rise; see check coverage."
                 if curve["observed_fall_and_rise"] else
                 "UNVERIFIED: available history has not established pressure falling and rising between dreams"),
        calibration="No automated threshold recommendation; inspect actual check coverage and source validity.",
        limits=["Do not interpolate missing checks or call cycle-to-cycle variation discharge/recovery.",
                "Manual and failed cycles remain visible; every attempt end affects the timer.",
                "new_counts distinguishes formula families, not every deployed revision.",
                "Timer/cadence and legacy cap are explicit assumptions; historical settings were not saved."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, nargs="?")
    parser.add_argument("--start", type=utc, required=True)
    parser.add_argument("--end", type=utc, required=True)
    parser.add_argument("--print-sql", action="store_true")
    parser.add_argument("--with-checks", action="store_true", help="export new check history after its migration")
    parser.add_argument("--interval-hours", type=float, default=6)
    parser.add_argument("--check-seconds", type=float, default=600)
    parser.add_argument("--legacy-cap", type=int, default=50)
    args = parser.parse_args()
    if args.print_sql:
        print(export_sql(args.start, args.end, with_checks=args.with_checks))
        return
    if not args.history:
        parser.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    print(json.dumps(report(rows, args.start, args.end, interval_hours=args.interval_hours,
                           check_seconds=args.check_seconds, legacy_cap=args.legacy_cap), indent=2))


if __name__ == "__main__":
    main()
