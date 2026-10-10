#!/usr/bin/env python3
"""Rest drive eval: how often would curiosity and outreach have eased off? Read-only.

Two lanes, both classifying with the production function
(orion.regulation.rest_drive.read_rest_drive) and judging tiredness with the
production reader rule (rest_drive_view: only a fresh `due` is tired):

  saved   -- every check the dream service actually saved
             (dream_pressure_observation, live since 2026-10-09 06:53 UTC).
             Real states, real clocks. Short: the table is new.
  replay  -- 14 days of the current (novelty, #2557) pressure recomputed at
             every 600 s check with measure_dream_pressure_crossings's own
             loaders, and a threshold-3 schedule simulated with the LIVE gate
             order (all-chat idle, 6 h minimum, 48 h overdue backstop). Counter-
             factual before #2557; validated exact against saved checks after.

Consumer impact, per day: real curiosity runs (curiosity_run_outcomes.completed_at)
and real endogenous outreach sends (endogenous_outreach_decisions, outreach and
not forced) that happened while the drive read tired AND inside the stretched
cooldown (base <= gap since the previous one < base x multiplier). Those are the
events the drive would have held back. First-order only: a held event would have
shifted the ones after it, which this does not simulate.

Usage:
  python scripts/analysis/measure_rest_drive_easing.py --print-sql --start S --end E > export.sql
  psql ... -Atf export.sql > rows.jsonl     (read-only transaction)
  python scripts/analysis/measure_rest_drive_easing.py rows.jsonl --start S --end E
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.regulation.rest_drive import read_rest_drive, rest_drive_view  # noqa: E402
from orion.schemas.dream_cycle import SleepPressureV1  # noqa: E402


def _crossings():
    name = "_measure_dream_pressure_crossings"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts/analysis/measure_dream_pressure_crossings.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def utc(value):
    return _crossings().utc(value)


def export_sql(start, end):
    start, end = utc(start), utc(end)
    base = _crossings().export_sql(start, end, with_checks=True, with_sources=True).rstrip()
    if not base.endswith("ROLLBACK;"):
        raise ValueError("expected the crossings export to end its read-only transaction with ROLLBACK")
    consumers = f"""
SELECT row_to_json(r) FROM (
 SELECT 'consumer' AS kind, 'curiosity' AS consumer, completed_at AS at
   FROM curiosity_run_outcomes
  WHERE completed_at >= '{start.isoformat()}'::timestamptz AND completed_at < '{end.isoformat()}'::timestamptz
 UNION ALL
 SELECT 'consumer', 'outreach', decided_at
   FROM endogenous_outreach_decisions
  WHERE outreach AND NOT forced
    AND decided_at >= '{start.isoformat()}'::timestamptz AND decided_at < '{end.isoformat()}'::timestamptz
) r ORDER BY at;
"""
    # Inside the same read-only transaction, before its ROLLBACK.
    return base[: -len("ROLLBACK;")] + consumers + "ROLLBACK;\n"


# --- classification ---------------------------------------------------------------


def _reading_for(pressure, *, now, threshold, last_start, last_end, min_interval_hours, lookback_hours,
                 has_candidates, source_errors=(), source_ref="replay"):
    p = SleepPressureV1(since=pressure["since"], computed_at=now, pressure=pressure["pressure"],
                        threshold=threshold, idle_required_minutes=45.0)
    overdue = last_start is None or now - last_start >= timedelta(hours=lookback_hours)
    return read_rest_drive(p, now=now, source_ref=source_ref, last_attempt_end=last_end,
                           min_interval_hours=min_interval_hours, overdue=overdue,
                           has_candidates=has_candidates, source_errors=list(source_errors))


def saved_timeline(rows):
    """(at, DriveReadingV1) per saved check, classified exactly as the producer would."""
    out = []
    for r in rows:
        if r.get("kind") != "pressure_check":
            continue
        o = r["observation"]
        now = utc(o["observed_at"])
        reading = o["reading"]
        last_start = utc(o["last_window_start"]) if o.get("last_window_start") else None
        last_end = utc(o["last_attempt_end"]) if o.get("last_attempt_end") else None
        out.append((now, _reading_for(
            dict(since=utc(reading["since"]), pressure=float(reading["pressure"])), now=now,
            threshold=float(reading["threshold"]), last_start=last_start, last_end=last_end,
            min_interval_hours=float(o["min_interval_hours"]), lookback_hours=float(o["lookback_hours"]),
            has_candidates=bool(reading.get("counts")), source_errors=o.get("source_errors") or (),
            source_ref=o["check_id"])))
    out.sort(key=lambda x: x[0])
    return out


def replay_timeline(rows, start, end, *, threshold=3.0, check_seconds=600.0, lookback_hours=48.0,
                    interval_hours=6.0, idle_required=45.0, cycle_minutes=6.0):
    """Threshold-3 schedule with the live gate order, and the reading at every check."""
    m = _crossings()
    start, end = utc(start), utc(end)
    items = m.prepare_items([r for r in rows if r.get("kind") == "source"])
    times = [x[0] for x in items]
    chat_all = sorted(utc(r["at"]) for r in rows if r.get("kind") == "chat")
    cycles = sorted((r for r in rows if r.get("kind") not in {"source", "chat", "pressure_check", "consumer"}),
                    key=lambda r: utc(r["started_at"]))
    good_starts = [utc(c["started_at"]) for c in cycles if c["status"] != "failed"]
    last_start = next((s for s in reversed(good_starts) if s <= start), None)
    last_end = last_start + timedelta(minutes=cycle_minutes) if last_start else None
    out, sleeps, t = [], [], start
    while t < end:
        floor = t - timedelta(hours=lookback_hours)
        since = max(floor, last_start) if last_start else floor
        value, counts, _ = m.pressure_at(items, times, t, since, lookback_hours)
        reading = _reading_for(dict(since=since, pressure=value), now=t, threshold=threshold,
                               last_start=last_start, last_end=last_end, min_interval_hours=interval_hours,
                               lookback_hours=lookback_hours, has_candidates=bool(counts))
        out.append((t, reading))
        idle = m._idle_minutes(chat_all, t)
        is_idle = idle is not None and idle >= idle_required
        refractory = last_end is not None and t - last_end < timedelta(hours=interval_hours)
        if not refractory and is_idle and reading.state == "due":
            sleeps.append(t)
            last_start, last_end = t, t + timedelta(minutes=cycle_minutes)
        t += timedelta(seconds=check_seconds)
    return out, sleeps


# --- per-day tallies --------------------------------------------------------------


def tired_minutes_per_day(timeline, *, max_age_sec=1800.0):
    """Minutes per UTC day a reader would have judged Orion tired. Each reading
    holds until the next one, but never past the reader's staleness bound."""
    per_day = defaultdict(float)
    states = defaultdict(Counter)
    for i, (at, reading) in enumerate(timeline):
        nxt = timeline[i + 1][0] if i + 1 < len(timeline) else at + timedelta(seconds=600)
        hold_until = min(nxt, at + timedelta(seconds=max_age_sec))
        states[at.date().isoformat()][reading.state] += 1
        if rest_drive_view(reading, now=at, max_age_sec=max_age_sec).verdict != "tired":
            continue
        cur = at
        while cur < hold_until:  # split across midnight
            day_end = datetime(cur.year, cur.month, cur.day, tzinfo=timezone.utc) + timedelta(days=1)
            seg = min(hold_until, day_end)
            per_day[cur.date().isoformat()] += (seg - cur).total_seconds() / 60
            cur = seg
    return {d: round(per_day.get(d, 0.0), 1) for d in sorted(states)}, {d: dict(c) for d, c in sorted(states.items())}


def held_events(timeline, events, *, base_sec, multiplier, max_age_sec=1800.0):
    """Per day: events total, events while tired, and events the drive would have
    held (tired AND inside the stretched-but-not-the-base cooldown; first order)."""
    times = [t for t, _ in timeline]
    total, during, held = Counter(), Counter(), Counter()
    examples = []
    prev = None
    for at in sorted(events):
        day = at.date().isoformat()
        total[day] += 1
        i = bisect.bisect_right(times, at) - 1
        tired = i >= 0 and rest_drive_view(timeline[i][1], now=at, max_age_sec=max_age_sec).verdict == "tired"
        during[day] += tired
        if tired and prev is not None:
            gap = (at - prev).total_seconds()
            if base_sec <= gap < base_sec * multiplier:
                held[day] += 1
                examples.append(dict(at=at.isoformat(), gap_sec=round(gap), state_ref=timeline[i][1].source_ref))
        prev = at
    days = sorted(total)
    return dict(per_day={d: dict(events=total[d], while_tired=during[d], held=held[d]) for d in days},
                events=sum(total.values()), while_tired=sum(during.values()), held=sum(held.values()),
                examples=examples[:10])


def evaluate(rows, start, end, *, curiosity_base_sec=1800.0, outreach_base_sec=2700.0, multiplier=2.0):
    start, end = utc(start), utc(end)
    consumers = defaultdict(list)
    for r in rows:
        if r.get("kind") == "consumer":
            consumers[r["consumer"]].append(utc(r["at"]))
    saved = saved_timeline(rows)
    replay, sleeps = replay_timeline(rows, start, end)
    out = {"start": start.isoformat(), "end": end.isoformat(), "multiplier": multiplier,
           "curiosity_base_cooldown_sec": curiosity_base_sec, "outreach_base_cooldown_sec": outreach_base_sec}
    for name, timeline in (("saved", saved), ("replay", replay)):
        minutes, states = tired_minutes_per_day(timeline)
        span = (timeline[0][0].isoformat(), timeline[-1][0].isoformat()) if timeline else (None, None)
        lo = timeline[0][0] if timeline else None
        hi = timeline[-1][0] + timedelta(seconds=600) if timeline else None
        in_span = {k: [t for t in v if lo and lo <= t < hi] for k, v in consumers.items()}
        out[name] = dict(
            checks=len(timeline), span=span, tired_minutes_per_day=minutes, states_per_day=states,
            tired_minutes_total=round(sum(minutes.values()), 1),
            curiosity=held_events(timeline, in_span.get("curiosity", []), base_sec=curiosity_base_sec,
                                  multiplier=multiplier),
            outreach=held_events(timeline, in_span.get("outreach", []), base_sec=outreach_base_sec,
                                 multiplier=multiplier),
        )
    out["replay"]["simulated_sleeps"] = len(sleeps)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history", type=Path, nargs="?")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--print-sql", action="store_true")
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--curiosity-base-sec", type=float, default=1800.0)
    parser.add_argument("--outreach-base-sec", type=float, default=2700.0)
    args = parser.parse_args()
    if args.print_sql:
        print(export_sql(args.start, args.end))
        return
    if not args.history:
        parser.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate(rows, args.start, args.end, multiplier=args.multiplier,
                              curiosity_base_sec=args.curiosity_base_sec,
                              outreach_base_sec=args.outreach_base_sec), indent=2))


if __name__ == "__main__":
    main()
