#!/usr/bin/env python3
"""Replay the SHIPPED arousal reducer over saved history (Temporal Self rev 4, order 3 eval).

Unlike ``measure_arousal_replay.py`` (#2576, an offline experiment with its own classifier),
this steps ``orion.regulation.arousal.classify_arousal`` itself, at the regulate node's own
cadence (one step every ``--tick-sec``, plus an immediate step at each Juniper turn), and builds
each step's inputs exactly as ``services/orion-durable-runs/app/regulation_store.py`` does:

* E1 -- the newest Juniper turn at or before the step (``orion.regulation.juniper_turns`` rule,
  applied in the export SQL);
* S1 -- ``read_cabinet_heat`` over the last hour of athena cabinet readings, seeded with the
  previous step's thermal state and critical latch;
* S2 -- ``gpu_queue_evidence`` over the saved ``gpu_pool_state_history`` snapshots in the sustain
  window plus 60 s, per host, most-strained fresh host wins.

Reports minutes per level per UTC day, level transitions per day, and strained exits per day
(spec R3a: more than 12 strained -> engaged/idle exits a day means the clear time is too short).

Usage (read-only export, then replay):
    python3 scripts/analysis/measure_arousal_reducer_replay.py --start 2026-10-09T07:00:00+00:00 \
        --end 2026-10-10T18:00:00+00:00 --print-sql > /tmp/x.sql
    docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -At < /tmp/x.sql > /tmp/rows.jsonl
    python3 scripts/analysis/measure_arousal_reducer_replay.py rows.jsonl --start ... --end ...
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orion.autonomy.cabinet_heat import read_cabinet_heat  # noqa: E402
from orion.hardware_watch.rules import TempPoint  # noqa: E402
from orion.regulation.arousal import (  # noqa: E402
    DEFAULT_CLEAR_SEC,
    DEFAULT_ENGAGED_MINUTES,
    DEFAULT_GPU_QUEUE_FLOOR,
    DEFAULT_GPU_SUSTAIN_SEC,
    GPU_STATE_STALE_SEC,
    classify_arousal,
    gpu_queue_evidence,
)
from orion.regulation.juniper_turns import JUNIPER_TURN_PREDICATE  # noqa: E402
from orion.schemas.regulation import ArousalInputsV1  # noqa: E402

LEVELS = ("engaged", "idle", "strained", "unknown")
CABINET_LOOKBACK = timedelta(hours=1)
GPU_MARGIN_SEC = 60.0


def utc(value) -> datetime:
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def export_sql(start: datetime, end: datetime) -> str:
    start, end = utc(start), utc(end)
    if end <= start:
        raise ValueError("end must follow start")
    warm = start - timedelta(hours=1)
    lo, hi = warm.isoformat(), end.isoformat()
    return f"""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '120s';
SET LOCAL TIME ZONE 'UTC';
SELECT row_to_json(r) FROM (
 SELECT 'turn' AS kind, created_at AT TIME ZONE 'UTC' AS at FROM chat_history_log
 WHERE {JUNIPER_TURN_PREDICATE} AND created_at < '{hi}'::timestamptz
 AND (created_at >= '{lo}'::timestamptz OR id IN (SELECT id FROM chat_history_log
      WHERE {JUNIPER_TURN_PREDICATE} AND created_at < '{lo}'::timestamptz ORDER BY created_at DESC LIMIT 1))
 ORDER BY created_at
) r;
SELECT row_to_json(r) FROM (
 SELECT 'heat' AS kind, timestamp::timestamptz AS at, (measurements->>'cabinet_temp_c')::float AS temp_c
 FROM orion_biometrics_summary WHERE node='athena'
 AND timestamp >= '{warm.strftime('%Y-%m-%d %H:%M:%S')}' AND timestamp < '{end.strftime('%Y-%m-%d %H:%M:%S')}'
 AND measurements ? 'cabinet_temp_c' ORDER BY timestamp
) r;
SELECT row_to_json(r) FROM (
 SELECT 'gpu_state' AS kind, generated_at AS at, host, queue_depth FROM gpu_pool_state_history
 WHERE generated_at >= '{lo}'::timestamptz AND generated_at < '{hi}'::timestamptz ORDER BY host, generated_at
) r;
ROLLBACK;"""


def replay(rows, start, end, *, tick_sec=120.0, engaged_minutes=DEFAULT_ENGAGED_MINUTES,
           floor=DEFAULT_GPU_QUEUE_FLOOR, sustain_sec=DEFAULT_GPU_SUSTAIN_SEC, clear_sec=DEFAULT_CLEAR_SEC):
    start, end = utc(start), utc(end)
    turns = sorted(utc(r["at"]) for r in rows if r["kind"] == "turn")
    heat = sorted((utc(r["at"]), float(r["temp_c"])) for r in rows
                  if r["kind"] == "heat" and r.get("temp_c") is not None)
    heat_ts = [h[0] for h in heat]
    gpu: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["kind"] == "gpu_state":
            gpu[r["host"]].append((utc(r["at"]), r["queue_depth"]))
    gpu_ts = {h: [s[0] for s in sorted(v, key=lambda s: s[0])] for h, v in gpu.items()}
    gpu = {h: sorted(v, key=lambda s: s[0]) for h, v in gpu.items()}

    # The node's own step times: a tick every tick_sec from start, plus one at each Juniper turn.
    steps, t = [], start
    while t < end:
        steps.append(t)
        t += timedelta(seconds=tick_sec)
    steps = sorted(set(steps) | {x for x in turns if start <= x < end})

    prev = prev_inputs = None
    readings = []
    for now in steps:
        i = bisect.bisect_right(turns, now)
        minutes = (now - turns[i - 1]).total_seconds() / 60.0 if i else None
        lo = bisect.bisect_left(heat_ts, now - CABINET_LOOKBACK)
        hi = bisect.bisect_right(heat_ts, now)
        points = [TempPoint(ts=ts, value=v) for ts, v in heat[lo:hi]]
        seed = None
        if prev_inputs is not None and (now - prev_inputs.observed_at).total_seconds() > 3 * tick_sec:
            prev_inputs = None
        if prev_inputs is not None and prev_inputs.cabinet_thermal_state not in (None, "unknown"):
            seed = dataclasses.replace(read_cabinet_heat([], now), thermal_state=prev_inputs.cabinet_thermal_state,
                                       critical=prev_inputs.cabinet_critical)
        verdict = read_cabinet_heat(points, now, previous=seed)
        best = None
        for host, snaps in gpu.items():
            a = bisect.bisect_left(gpu_ts[host], now - timedelta(seconds=sustain_sec + GPU_MARGIN_SEC))
            b = bisect.bisect_right(gpu_ts[host], now)
            age, depth, sustained = gpu_queue_evidence(snaps[a:b], now, floor=floor)
            if age is None:
                continue
            key = (age <= GPU_STATE_STALE_SEC, sustained or 0.0, -age)
            if best is None or key > best[0]:
                best = (key, age, depth, sustained)
        inputs = ArousalInputsV1(
            observed_at=now, juniper_turn_read_ok=True, minutes_since_juniper_turn=minutes,
            cabinet_read_ok=True, cabinet_reflex=verdict.reflex, cabinet_thermal_state=verdict.thermal_state,
            cabinet_critical=verdict.critical, cabinet_temp_c=verdict.temp_c,
            **({} if best is None else {"gpu_state_age_sec": best[1], "gpu_queue_depth": best[2],
                                        "gpu_queue_sustained_sec": best[3]}))
        prev = classify_arousal(prev, inputs, engaged_minutes=engaged_minutes, gpu_queue_floor=floor,
                                gpu_sustain_sec=sustain_sec, clear_sec=clear_sec, max_prev_gap_sec=3 * tick_sec)
        prev_inputs = inputs
        readings.append(prev)
    return summarize(readings, end)


def summarize(readings, end):
    minutes = defaultdict(lambda: {k: 0.0 for k in LEVELS})
    transitions, strained_exits, reasons_when_unknown = Counter(), Counter(), Counter()
    for cur, nxt in zip(readings, readings[1:] + [None]):
        a = cur.observed_at
        b = nxt.observed_at if nxt is not None else end
        while a < b:   # split a hold across UTC midnights
            cut = min(b, datetime.combine(a.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc))
            minutes[a.date().isoformat()][cur.arousal_level] += (cut - a).total_seconds() / 60.0
            a = cut
        if cur.arousal_level == "unknown":
            reasons_when_unknown.update(cur.reasons)
        if nxt is not None and nxt.arousal_level != cur.arousal_level:
            day = nxt.observed_at.date().isoformat()
            transitions[day] += 1
            if cur.arousal_level == "strained" and nxt.arousal_level in ("engaged", "idle"):
                strained_exits[day] += 1
    totals = {k: round(sum(d[k] for d in minutes.values()), 1) for k in LEVELS}
    known = sum(totals[k] for k in ("engaged", "idle", "strained"))
    shares = {k: round(totals[k] / known, 4) for k in ("engaged", "idle", "strained")} if known else None
    return {
        "steps": len(readings),
        "minutes_per_day": {d: {k: round(v, 1) for k, v in m.items()} for d, m in sorted(minutes.items())},
        "total_minutes": totals,
        "share_of_known": shares,
        "acceptance_bar_every_level_between_0_and_100pct": bool(shares) and all(0 < v < 1 for v in shares.values()),
        "transitions_per_day": dict(sorted(transitions.items())),
        "strained_exits_per_day": dict(sorted(strained_exits.items())),
        "hysteresis_review_days": [d for d, n in strained_exits.items() if n > 12],
        "unknown_step_reasons": dict(reasons_when_unknown),
        "strained_spans": _spans(readings, end, "strained"),
    }


def _spans(readings, end, level):
    out, open_at = [], None
    for cur, nxt in zip(readings, readings[1:] + [None]):
        if cur.arousal_level == level and open_at is None:
            open_at = cur.observed_at
        closes = nxt is None or nxt.arousal_level != level
        if open_at is not None and closes:
            stop = nxt.observed_at if nxt is not None else end
            out.append({"from": open_at.isoformat(), "to": stop.isoformat(),
                        "minutes": round((stop - open_at).total_seconds() / 60.0, 1),
                        "reasons": sorted({r for r in cur.reasons})})
            open_at = None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("history", type=Path, nargs="?")
    p.add_argument("--start", type=utc, required=True)
    p.add_argument("--end", type=utc, required=True)
    p.add_argument("--tick-sec", type=float, default=120.0)
    p.add_argument("--gpu-queue-floor", type=int, default=DEFAULT_GPU_QUEUE_FLOOR)
    p.add_argument("--print-sql", action="store_true")
    args = p.parse_args()
    if args.print_sql:
        print(export_sql(args.start, args.end))
        return
    if not args.history:
        p.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip().startswith("{")]
    result = replay(rows, args.start, args.end, tick_sec=args.tick_sec, floor=args.gpu_queue_floor)
    result.update(start=args.start.isoformat(), end=args.end.isoformat(), tick_sec=args.tick_sec,
                  gpu_queue_floor=args.gpu_queue_floor, input_counts=dict(Counter(r["kind"] for r in rows)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
