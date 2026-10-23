#!/usr/bin/env python3
"""R3 readers eval: how often would each arousal reader have changed its decision? Read-only.

Arousal timeline, two lanes:

  replay -- the SHIPPED reducer (``classify_arousal``) stepped over saved inputs by
            ``measure_arousal_reducer_replay.replay_readings`` (Juniper turns, athena cabinet
            readings, ``gpu_pool_state_history``). Covers the whole window, but only as far
            back as ``gpu_pool_state_history`` goes (10-09 06:46 UTC): before that S2 is
            missing and the replay reads ``unknown``.
  live   -- the ``arousal_transition`` rows the regulate node actually wrote to
            ``temporal_self_event`` (live since 10-10 23:00 UTC). Each row opens a level at
            its ``occurred_at``; the level holds until the next row.

Decisions, each joined to the level in force at its own timestamp with the production reader
rules (the gate functions' own branches, restated here as a level check):

  dream       non-forced sleeps (``dream_cycle.trigger = 'pressure'``). Held if engaged or strained
              and that level began less than ``--dream-max-hold-hours`` (6) before the sleep.
  curiosity   non-urgent investigation-line runs (``durable_admission_runs``, workflow
              ``curiosity.investigate``, line ``investigate``). Held if strained, or if
              engaged and the gap since the previous run was under base cooldown x multiplier.
  self_inquiry  self-inquiry runs (same table, line ``self_inquiry``). Held unless idle.
  outreach    sent background outreach (``endogenous_outreach_decisions``: outreach, not
              Door-A ``curiosity_outreach``; forced debug sends included, since the shipped
              gate holds them too). Held if strained.
  visual_baseline  (reader 4) baseline visual-reverie attempts (``reverie_visual_attempt``
              with a ``need_id``, at ``started_at``). Held (deferred by retry_sec) if strained.
  reverie_chain  (reader 5) completed text-reverie chains (``substrate_reverie_chain.created_at``
              = when the pause before the next chain starts). Held = that pause doubled if
              strained.
  mutation    (reader 6) self-modification proposals (``substrate_mutation_proposal``).
              Held if strained. Includes any operator-triggered manual-cycle proposals (not
              distinguishable in the row), which the shipped reader never holds: upper bound.

Unknown never holds anything. First order only: a held decision would have moved the ones
after it, which this does not simulate. Curiosity rows do not record whether a run was forced
(a forced run would not have been held), so curiosity counts are an upper bound.

Live-lane caveat: a level holds until the next ``arousal_transition`` row. If durable-runs
stopped writing, the real readers would read unknown after 360 s while this lane keeps the
last level, so the live lane can overcount holds across a durable-runs outage.

A GPU queue jammed by a refusing lane controller reads ``unknown`` (``gpu_fault:<kind>:<seat>``),
not strained (``orion.regulation.arousal.gpu_fault_from_events``). ``--no-gpu-fault`` replays
the reducer without that input, for a before/after on the same export. Every strained span is
listed with its reasons.

Usage:
  python3 scripts/analysis/measure_arousal_reader_replay.py --print-sql --start S --end E > x.sql
  docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -At < x.sql > rows.jsonl
  python3 scripts/analysis/measure_arousal_reader_replay.py rows.jsonl --start S --end E
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

READERS = ("dream", "curiosity", "self_inquiry", "outreach", "visual_baseline", "reverie_chain", "mutation")


def _reducer_replay():
    name = "_measure_arousal_reducer_replay"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts/analysis/measure_arousal_reducer_replay.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def utc(value):
    return _reducer_replay().utc(value)


def export_sql(start, end):
    start, end = utc(start), utc(end)
    base = _reducer_replay().export_sql(start, end).rstrip()
    if not base.endswith("ROLLBACK;"):
        raise ValueError("expected the reducer export to end its read-only transaction with ROLLBACK")
    lo, hi = start.isoformat(), end.isoformat()
    extra = f"""
SELECT row_to_json(r) FROM (
 SELECT 'decision' AS kind, 'dream' AS reader, started_at AS at FROM dream_cycle
  WHERE trigger = 'pressure' AND started_at >= '{lo}'::timestamptz AND started_at < '{hi}'::timestamptz
 UNION ALL
 SELECT 'decision', CASE WHEN request->'brief'->>'line' = 'self_inquiry' THEN 'self_inquiry' ELSE 'curiosity' END,
        created_at FROM durable_admission_runs
  WHERE request->>'workflow' = 'curiosity.investigate'
    AND COALESCE(request->'brief'->>'line', 'investigate') IN ('investigate', 'self_inquiry')
    AND COALESCE(request->'brief'->>'urgent', '') = ''
    AND created_at >= '{(start - timedelta(days=1)).isoformat()}'::timestamptz AND created_at < '{hi}'::timestamptz
 UNION ALL
 SELECT 'decision', 'outreach', decided_at FROM endogenous_outreach_decisions
  WHERE outreach AND COALESCE(result_json->>'source', '') <> 'curiosity_outreach'
    AND decided_at >= '{lo}'::timestamptz AND decided_at < '{hi}'::timestamptz
 UNION ALL
 SELECT 'decision', 'visual_baseline', started_at FROM reverie_visual_attempt
  WHERE need_id IS NOT NULL AND started_at >= '{lo}'::timestamptz AND started_at < '{hi}'::timestamptz
 UNION ALL
 SELECT 'decision', 'reverie_chain', created_at FROM substrate_reverie_chain
  WHERE created_at >= '{lo}'::timestamptz AND created_at < '{hi}'::timestamptz
 UNION ALL
 SELECT 'decision', 'mutation', created_at FROM substrate_mutation_proposal
  WHERE created_at >= '{lo}'::timestamptz AND created_at < '{hi}'::timestamptz
) r ORDER BY at;
SELECT row_to_json(r) FROM (
 SELECT 'live_level' AS kind, occurred_at AS at, label AS level FROM temporal_self_event
  WHERE source_kind = 'arousal_transition' AND occurred_at < '{hi}'::timestamptz
    -- Level changes only: strain cause/seed/keepalive rows (event id ``...:at:<time>``, strain loop
    -- part 1) repeat the level and would restart its age here.
    AND event_id NOT LIKE '%:at:%' ORDER BY occurred_at
) r;
"""
    return base[: -len("ROLLBACK;")] + extra + "ROLLBACK;\n"


class Timeline:
    """Level in force at any instant: a sorted list of (start, level[, since])."""

    def __init__(self, points):
        self.points = sorted((p[0], p[1], p[2] if len(p) > 2 else p[0]) for p in points)
        self.ts = [p[0] for p in self.points]

    def level_at(self, at):
        return self.level_and_since(at)[0]

    def level_and_since(self, at):
        i = bisect.bisect_right(self.ts, at)
        return (self.points[i - 1][1], self.points[i - 1][2]) if i else ("unknown", None)


def held(reader, level, gap_sec, *, curiosity_base_sec, engaged_multiplier,
         level_age_sec=None, dream_max_hold_sec=6 * 3600.0):
    """Would this reader's gate have refused at `level`? Mirrors the shipped branches."""
    if level in (None, "unknown", "idle"):
        return None
    if reader == "dream":
        if level_age_sec is not None and level_age_sec >= dream_max_hold_sec:
            return None  # DREAM_AROUSAL_MAX_HOLD_HOURS: a long-held level stops holding sleep
        return f"arousal_{level}"
    if reader == "self_inquiry":
        return f"arousal_{level}"
    if reader in ("outreach", "visual_baseline", "reverie_chain", "mutation"):
        return "arousal_strained" if level == "strained" else None
    if reader == "curiosity":
        if level == "strained":
            return "arousal_strained"
        if engaged_multiplier > 1.0 and gap_sec is not None and gap_sec < curiosity_base_sec * engaged_multiplier:
            return "arousal_engaged_cooldown"
    return None


def reader_impact(decisions, timeline, start, end, **kw):
    per_day = defaultdict(lambda: {r: {"decisions": 0, "held": 0} for r in READERS})
    reasons = defaultdict(Counter)
    examples = defaultdict(list)
    last_at = {}
    for reader, at in sorted(decisions, key=lambda d: d[1]):
        gap = None if reader not in last_at else (at - last_at[reader]).total_seconds()
        last_at[reader] = at
        if not (start <= at < end):
            continue  # earlier runs only seed the curiosity gap
        day = at.date().isoformat()
        per_day[day][reader]["decisions"] += 1
        level, since = timeline.level_and_since(at)
        age = None if since is None else (at - since).total_seconds()
        why = held(reader, level, gap, level_age_sec=age, **kw)
        if why:
            per_day[day][reader]["held"] += 1
            reasons[reader][why] += 1
            if len(examples[reader]) < 5:
                examples[reader].append({"at": at.isoformat(), "level": level, "reason": why,
                                         "gap_min": None if gap is None else round(gap / 60.0, 1)})
    return {
        "per_day": {d: v for d, v in sorted(per_day.items())},
        "totals": {r: {"decisions": sum(v[r]["decisions"] for v in per_day.values()),
                       "held": sum(v[r]["held"] for v in per_day.values())} for r in READERS},
        "held_reasons": {r: dict(c) for r, c in reasons.items()},
        "examples": dict(examples),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("history", type=Path, nargs="?")
    p.add_argument("--start", type=utc, required=True)
    p.add_argument("--end", type=utc, required=True)
    p.add_argument("--print-sql", action="store_true")
    # Live Hub values (services/orion-hub/.env, 10-12): 1800 s floor, cap 7, no window ->
    # paced cooldown = the floor. Engaged multiplier 2.0 (HUB_CURIOSITY_AROUSAL_*).
    p.add_argument("--curiosity-base-cooldown-sec", type=float, default=1800.0)
    p.add_argument("--engaged-multiplier", type=float, default=2.0)
    p.add_argument("--dream-max-hold-hours", type=float, default=6.0)
    p.add_argument("--no-gpu-fault", action="store_true", help="replay without the GPU fault input (before)")
    args = p.parse_args()
    if args.print_sql:
        print(export_sql(args.start, args.end))
        return
    if not args.history:
        p.error("history export required")
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip().startswith("{")]
    decisions = [(r["reader"], utc(r["at"])) for r in rows if r.get("kind") == "decision"]
    kw = dict(curiosity_base_sec=args.curiosity_base_cooldown_sec, engaged_multiplier=args.engaged_multiplier,
              dream_max_hold_sec=args.dream_max_hold_hours * 3600.0)

    rr = _reducer_replay()
    readings = rr.replay_readings(rows, args.start, args.end, gpu_fault=not args.no_gpu_fault)
    replay_tl = Timeline((r.observed_at, r.arousal_level, r.since) for r in readings)
    summary = rr.summarize(readings, args.end)
    out = {
        "start": args.start.isoformat(), "end": args.end.isoformat(),
        "input_counts": dict(Counter(r["kind"] for r in rows)),
        "decision_counts": dict(Counter(r for r, _ in decisions)),
        "replay": {
            "level_minutes_per_day": summary["minutes_per_day"],
            "strained_spans": summary["strained_spans"],
            "unknown_spans": summary["unknown_spans"],
            "readers": reader_impact(decisions, replay_tl, args.start, args.end, **kw),
        },
    }
    live = [(utc(r["at"]), r["level"]) for r in rows if r.get("kind") == "live_level"]
    if live:
        live_start = max(args.start, live[0][0])
        out["live"] = {
            "from": live_start.isoformat(),
            "transitions": [{"at": a.isoformat(), "level": lv} for a, lv in live if a >= live_start],
            "readers": reader_impact(decisions, Timeline(live), live_start, args.end, **kw),
        }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
