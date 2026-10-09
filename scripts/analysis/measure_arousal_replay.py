#!/usr/bin/env python3
"""Replay PR #2369 Rev 4 labels offline, with hourly evidence and explicit unknowns.

No text leaves chat storage: export only turn timestamps and the proposed E1
predicate. Outreach and focus annotate hours but NEVER vote for arousal.
GPU state snapshots are not persisted by the current SQL writer. Lease events
support a separate PROVISIONAL replay; they cannot prove 5-second freshness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orion.autonomy.cabinet_heat import read_cabinet_heat, REFLEX_CABINET_HOT
from orion.hardware_watch.rules import TempPoint

LEVELS = ("engaged", "idle", "strained", "unknown")
GPU_EXITS = {"queued", "granted", "released", "cancelled", "aborted", "expired",
             "dead_lettered", "unavailable", "recalled"}


def utc(value):
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def export_sql(start, end):
    start, end = utc(start), utc(end)
    if end <= start:
        raise ValueError("end must follow start")
    warm = start - timedelta(hours=1)
    lo, hi = warm.isoformat(), end.isoformat()
    return f"""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL TIME ZONE 'UTC';
SELECT row_to_json(r) FROM (
 SELECT 'chat' AS kind, created_at AT TIME ZONE 'UTC' AS at,
        source = 'hub_orion' AND btrim(coalesce(prompt,'')) <> '' AS juniper
 FROM chat_history_log WHERE created_at < '{hi}'::timestamptz
 AND (created_at >= '{lo}'::timestamptz OR id IN (
   SELECT id FROM chat_history_log WHERE created_at < '{lo}'::timestamptz ORDER BY created_at DESC LIMIT 1
 ) OR id IN (
   SELECT id FROM chat_history_log WHERE created_at < '{lo}'::timestamptz
     AND source='hub_orion' AND btrim(coalesce(prompt,''))<>'' ORDER BY created_at DESC LIMIT 1
 )) ORDER BY created_at
) r;
SELECT row_to_json(r) FROM (
 SELECT 'heat' AS kind, timestamp::timestamptz AS at,
        (measurements->>'cabinet_temp_c')::float AS temp_c
 FROM orion_biometrics_summary WHERE node='athena'
 AND timestamp >= '{warm.strftime('%Y-%m-%d %H:%M:%S')}'
 AND timestamp < '{end.strftime('%Y-%m-%d %H:%M:%S')}'
 AND measurements ? 'cabinet_temp_c' ORDER BY timestamp
) r;
SELECT row_to_json(r) FROM (
 SELECT 'gpu_event' AS kind, generated_at AS at, event, lease_id
 FROM gpu_pool_events WHERE generated_at >= '{lo}'::timestamptz AND generated_at < '{hi}'::timestamptz
 AND lease_id IN (SELECT lease_id FROM gpu_pool_events WHERE event='backlogged'
   AND generated_at >= '{lo}'::timestamptz AND generated_at < '{hi}'::timestamptz)
 ORDER BY generated_at, event_id
) r;
SELECT row_to_json(r) FROM (
 SELECT 'outreach' AS kind, decided_at AS at FROM endogenous_outreach_decisions
 WHERE outreach AND decided_at >= '{lo}'::timestamptz AND decided_at < '{hi}'::timestamptz
) r;
SELECT row_to_json(r) FROM (
 SELECT 'focus' AS kind, started_at AS at, ended_at, target_id FROM field_dominance_run
 WHERE started_at < '{hi}'::timestamptz AND ended_at > '{lo}'::timestamptz
) r;
ROLLBACK;"""


class Classifier:
    """Experiment only. Never imported by a running service."""
    def __init__(self):
        self.level = "unknown"
        self.strain_latched = False
        self.clear_since = None
        self.backlog_since = None

    def step(self, at, *, hot, backlog, last_turn, idle_minutes=45):
        if backlog is True:
            if self.backlog_since is None:
                self.backlog_since = at
        else:
            self.backlog_since = None
        gpu_strain = self.backlog_since is not None and (at-self.backlog_since).total_seconds() >= 300
        if hot is True or gpu_strain:
            self.strain_latched = True
            self.clear_since = None
            self.level = "strained"
        elif hot is None or backlog is None or last_turn is None:
            self.clear_since = None  # missing time never earns clear minutes
            self.level = "unknown"
        elif self.strain_latched:
            if hot is False and backlog is False:
                if self.clear_since is None:
                    self.clear_since = at
                if (at-self.clear_since).total_seconds() >= 600:
                    self.strain_latched = False
            else:
                self.clear_since = None
            self.level = "strained" if self.strain_latched else (
                "engaged" if (at-last_turn).total_seconds() < idle_minutes*60 else "idle")
        else:
            self.level = "engaged" if (at-last_turn).total_seconds() < idle_minutes*60 else "idle"
        return self.level


def replay(rows, start, end, *, idle_minutes=45):
    start, end = utc(start), utc(end)
    if end <= start or not math.isfinite(idle_minutes) or idle_minutes < 0:
        raise ValueError("invalid window or idle threshold")
    parsed = sorted([(utc(r["at"]), r) for r in rows], key=lambda p: p[0])
    allowed = {"chat", "heat", "gpu_event", "gpu_state", "focus", "outreach"}
    if any(r["kind"] not in allowed for _, r in parsed):
        raise ValueError("unknown input kind")
    strict, provisional, all_chat = Classifier(), Classifier(), Classifier()
    warm = start - timedelta(hours=1)
    at, i = warm, 0
    last_turn = last_chat = latest_heat = latest_gpu = None
    heat_verdict = None
    backlogged = set()
    hourly, transitions, exits = {}, [], Counter()
    previous = None
    while at < end:
        while i < len(parsed) and parsed[i][0] <= at:
            ts, row = parsed[i]
            kind = row["kind"]
            if kind == "chat":
                last_chat = ts
                if row["juniper"] is True:
                    last_turn = ts
            elif kind == "heat":
                temp = row["temp_c"]
                # Missing/non-finite samples do not refresh the last good observation.
                if temp is not None and math.isfinite(float(temp)):
                    latest_heat = TempPoint(ts=ts, value=float(temp))
                    heat_verdict = read_cabinet_heat([latest_heat], ts, previous=heat_verdict)
            elif kind == "gpu_state":
                depth = row["backlog_depth"]
                if not isinstance(depth, dict) or any(not isinstance(n, int) or n < 0 for n in depth.values()):
                    raise ValueError("invalid backlog_depth")
                latest_gpu = (ts, sum(depth.values()) > 0)
            elif kind == "gpu_event":
                if row["event"] == "backlogged":
                    backlogged.add(row["lease_id"])
                elif row["event"] in GPU_EXITS:
                    backlogged.discard(row["lease_id"])
            i += 1
        # S1 30-second producer cadence; S2 5-second cadence. Neither a
        # heartbeat nor a lease event is a saved GpuPoolStateV1 observation.
        hot = (heat_verdict.reflex == REFLEX_CABINET_HOT
               if latest_heat and (at-latest_heat.ts).total_seconds() <= 90 else None)
        backlog = latest_gpu[1] if latest_gpu and (at-latest_gpu[0]).total_seconds() <= 15 else None
        label = strict.step(at, hot=hot, backlog=backlog, last_turn=last_turn, idle_minutes=idle_minutes)
        approx = provisional.step(at, hot=hot, backlog=bool(backlogged), last_turn=last_turn, idle_minutes=idle_minutes)
        contaminated = all_chat.step(at, hot=hot, backlog=bool(backlogged), last_turn=last_chat, idle_minutes=idle_minutes)
        next_at = min(at+timedelta(seconds=5), end,
                      at.replace(minute=0, second=0, microsecond=0)+timedelta(hours=1))
        if at >= start:
            hour = at.replace(minute=0, second=0, microsecond=0).isoformat()
            bucket = hourly.setdefault(hour, dict(hour=hour, seconds=0,
                strict_seconds={k: 0 for k in LEVELS}, provisional_seconds={k: 0 for k in LEVELS},
                all_chat_provisional_seconds={k: 0 for k in LEVELS}, gpu_state_missing_seconds=0,
                cabinet_missing_seconds=0, event_reconstructed_backlog_seconds=0,
                juniper_turns=0, other_chat_rows=0, outreach_sent=0, focus_wall_span_seconds={}))
            seconds = (next_at-at).total_seconds()
            bucket["seconds"] += seconds
            bucket["strict_seconds"][label] += seconds
            bucket["provisional_seconds"][approx] += seconds
            bucket["all_chat_provisional_seconds"][contaminated] += seconds
            bucket["gpu_state_missing_seconds"] += seconds if backlog is None else 0
            bucket["cabinet_missing_seconds"] += seconds if hot is None else 0
            bucket["event_reconstructed_backlog_seconds"] += seconds if backlogged else 0
            if previous != (label, approx):
                transitions.append(dict(at=at.isoformat(), strict=label, provisional=approx))
                if previous and previous[1] == "strained" and approx in {"idle", "engaged"}:
                    exits[at.date().isoformat()] += 1
                previous = (label, approx)
        at = min(next_at, start) if at < start else next_at
    for ts, row in parsed:
        if row["kind"] == "focus":
            finish = utc(row["ended_at"])
            if finish < ts:
                raise ValueError("focus ends before it starts")
            for hour, bucket in hourly.items():
                lo = max(utc(hour), start, ts)
                hi = min(utc(hour)+timedelta(hours=1), end, finish)
                if hi > lo:
                    spans = bucket["focus_wall_span_seconds"]
                    spans[row["target_id"]] = spans.get(row["target_id"], 0)+(hi-lo).total_seconds()
        elif start <= ts < end and row["kind"] in {"chat", "outreach"}:
            bucket = hourly[ts.replace(minute=0, second=0, microsecond=0).isoformat()]
            key = "outreach_sent" if row["kind"] == "outreach" else (
                "juniper_turns" if row["juniper"] is True else "other_chat_rows")
            bucket[key] += 1
    totals = {lane: {level: sum(b[lane][level] for b in hourly.values()) for level in LEVELS}
              for lane in ("strict_seconds", "provisional_seconds", "all_chat_provisional_seconds")}
    return dict(start=start.isoformat(), end=end.isoformat(), hours=list(hourly.values()),
        totals=totals, transitions=transitions, provisional_strain_exits_per_day=dict(exits),
        hysteresis_review_days=[day for day, count in exits.items() if count > 12],
        input_counts=dict(Counter(r["kind"] for _, r in parsed)), idle_minutes_assumed=idle_minutes,
        verdict="UNVERIFIED: provisional labels require human comparison and complete GPU state history",
        assumptions=["UTC; five-second grid (transitions may lag by <5s); one-hour warmup.",
                     "Chat query success is evidence of freshness; age of the last turn is not sensor age.",
                     "Strict GPU state is unknown without snapshots <=15s old.",
                     "Provisional GPU starts clear at warmup and assumes complete ordered lease transitions.",
                     "Heat uses the current pure cabinet verdict; historical AC-low fallback is unavailable.",
                     "Outreach and focus annotate hours, never change labels; focus spans may include downtime.",
                     "All-chat comparison includes every chat row, isolating the old idle-query contamination."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, nargs="?")
    parser.add_argument("--start", type=utc, required=True)
    parser.add_argument("--end", type=utc, required=True)
    parser.add_argument("--idle-minutes", type=float, default=45)
    parser.add_argument("--print-sql", action="store_true")
    args = parser.parse_args()
    if args.print_sql:
        print(export_sql(args.start, args.end))
        return
    if not args.history:
        parser.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    print(json.dumps(replay(rows, args.start, args.end, idle_minutes=args.idle_minutes), indent=2))


if __name__ == "__main__":
    main()
