#!/usr/bin/env python3
"""Replay stored telemetry_anomaly windows through the episode gate.

Two sources (--source):

* ``triggers`` (default until enough scores accumulate): every pre-episode
  telemetry_anomaly row in metacog_trigger, one per flagged 60s window. Rows
  the episode gate itself published (upstream carries ``episode_id``) are
  excluded, so a post-deploy export cannot replay its own output. Caveat: the
  old shared 30s cooldown occasionally dropped a flagged window when another
  trigger kind held the lane; those read as calm gaps here and can split an
  episode, so "after" counts from this source are an upper bound.
* ``scores``: field_channel_anomaly_score (written by orion-sql-writer from
  2026-10-13), every window flagged or not, flagged = recon_loss >
  recon_error_p95 * --multiplier. Exact; use once it covers the period.

Before = flagged windows (one fire each under the shipped behaviour).
After  = the same windows folded through orion.metacog.anomaly_episode with
harness turns (harness_turn_trace: [created_at - fcc_elapsed_sec, created_at])
fed through TurnActivityTracker in time order, as the live gate sees them.

Read-only. Two steps, so no DB driver is needed:

  python3 scripts/analysis/replay_telemetry_anomaly_episodes.py --export-sql --days 15 \\
    | docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -At > /tmp/ta.jsonl
  python3 scripts/analysis/replay_telemetry_anomaly_episodes.py --rows /tmp/ta.jsonl

Dream novelty: orion-dream counts NEW distinct metacog keys (trigger_reason with
numbers -> '#', services/orion-dream/app/cycle_store.py) among degraded/critical
rows. For telemetry_anomaly that key is direction + lead channel, so the report
counts distinct (direction, lead) per day at ratio >= 1.5 (evidence_map's
degraded band) before and after.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orion.metacog.anomaly_episode import (  # noqa: E402
    DEFAULT_CALM_GAP_SEC,
    DEFAULT_SUSTAINED_HORIZON_SEC,
    DEFAULT_SUSTAINED_WINDOWS,
    DEFAULT_TURN_CARRY_SEC,
    AnomalyEpisodeStateV1,
    TurnActivityTracker,
    mark_published,
    step,
)
from orion.metacog.evidence_map import TELEMETRY_RATIO_CRITICAL, TELEMETRY_RATIO_DEGRADED  # noqa: E402


TURN_ACTIVITY_STEP_SEC = 300.0


def export_sql(days: int, source: str = "triggers", multiplier: float = 3.0) -> str:
    days = int(days)
    if days <= 0 or days > 60:
        raise ValueError("days must be 1..60")
    multiplier = float(multiplier)
    if source == "triggers":
        windows = f"""SELECT row_to_json(r) FROM (
 SELECT 'fire' AS kind, upstream->>'window_start' AS window_start,
        upstream->>'window_end' AS window_end,
        (upstream->>'recon_loss')::float / NULLIF((upstream->>'threshold')::float, 0) AS ratio,
        upstream->>'deviation_direction' AS direction,
        split_part(upstream->'top_channels'->>0, '=', 1) AS lead
 FROM metacog_trigger
 WHERE trigger_kind = 'telemetry_anomaly' AND timestamp > now() - interval '{days} days'
   AND NOT (upstream ? 'episode_id')
 ORDER BY timestamp
) r;"""
    elif source == "scores":
        # DISTINCT ON window: a stalled digester buffer can rescore one window.
        windows = f"""SELECT row_to_json(r) FROM (
 SELECT DISTINCT ON (encoder_id, window_end) 'score' AS kind, window_start, window_end,
        recon_loss / NULLIF(recon_error_p95 * {multiplier}, 0) AS ratio,
        deviation_direction AS direction,
        split_part(top_channels->>0, '=', 1) AS lead
 FROM field_channel_anomaly_score
 WHERE window_end > now() - interval '{days} days'
 ORDER BY encoder_id, window_end
) r;"""
    else:
        raise ValueError("source must be triggers or scores")
    return f"""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL TIME ZONE 'UTC';
{windows}
SELECT row_to_json(r) FROM (
 SELECT 'turn' AS kind, correlation_id AS cid, created_at AS ended_at,
        (run_artifact->>'fcc_elapsed_sec')::float AS elapsed_sec
 FROM harness_turn_trace
 WHERE created_at > now() - interval '{days + 1} days'
 ORDER BY created_at
) r;
COMMIT;
"""


def _ts(value: str) -> float:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:  # harness_turn_trace.created_at is naive UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def load_rows(lines: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def lead_category(lead: str, _cache: dict[str, str | None] = {}) -> str | None:
    if lead not in _cache:
        from orion.field.channel_glossary import resolve_channel_entry

        entry = resolve_channel_entry(lead) if lead else None
        _cache[lead] = entry.category if entry else None
    return _cache[lead]


def compact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixture form: timestamps as epoch seconds, glossary category resolved."""
    out = []
    for r in rows:
        if r.get("kind") in ("fire", "score") and r.get("window_end") and r.get("ratio") is not None:
            if r["kind"] == "score" and float(r["ratio"]) <= 1.0:
                out.append({"k": "f", "ws": round(_ts(r["window_start"]), 3),
                            "we": round(_ts(r["window_end"]), 3), "r": round(float(r["ratio"]), 4),
                            "d": "", "l": "", "c": None, "calm": True})
                continue
            out.append({
                "k": "f", "ws": round(_ts(r["window_start"]), 3), "we": round(_ts(r["window_end"]), 3),
                "r": round(float(r["ratio"]), 4), "d": r.get("direction") or "mixed",
                "l": r.get("lead") or "", "c": lead_category(r.get("lead") or ""),
            })
        elif r.get("kind") == "turn" and r.get("ended_at"):
            out.append({"k": "t", "cid": r.get("cid") or "", "end": round(_ts(r["ended_at"]), 3),
                        "el": r.get("elapsed_sec")})
    return out


def _dream_weight(ratio: float) -> float:
    """orion-dream SEVERITY_WEIGHT for the evidence_map band of this ratio (0 = not counted)."""
    if ratio >= TELEMETRY_RATIO_CRITICAL:
        return 1.0
    if ratio >= TELEMETRY_RATIO_DEGRADED:
        return 0.6
    return 0.0


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat()


def replay(
    items: list[dict[str, Any]],
    *,
    calm_gap_sec: float = DEFAULT_CALM_GAP_SEC,
    turn_carry_sec: float = DEFAULT_TURN_CARRY_SEC,
    turn_suppress: bool = True,
    sustained_windows: int = DEFAULT_SUSTAINED_WINDOWS,
    sustained_horizon_sec: float = DEFAULT_SUSTAINED_HORIZON_SEC,
) -> dict[str, Any]:
    """`items` in compact() form. Returns per-day before/after counts + every after-fire."""
    events: list[tuple[float, int, dict[str, Any]]] = []
    for it in items:
        if it["k"] == "f":
            events.append((it["we"], 2, it))
        else:
            el = it.get("el")
            start = it["end"] - float(el) if el else it["end"]
            events.append((start, 0, {**it, "phase": "start"}))
            # Live, harness-governor grammar events stream through the turn
            # (measured longest in-turn silence: p99 840s, max 855s). The stored
            # trace keeps only start/end, so emit synthetic activity every
            # TURN_ACTIVITY_STEP_SEC to mirror what the live tracker sees.
            t = start + TURN_ACTIVITY_STEP_SEC
            while t < it["end"]:
                events.append((t, 0, {**it, "phase": "start"}))
                t += TURN_ACTIVITY_STEP_SEC
            events.append((it["end"], 1, {**it, "phase": "end"}))
    events.sort(key=lambda e: (e[0], e[1], e[2].get("we", 0.0)))

    tracker = TurnActivityTracker(turn_carry_sec=turn_carry_sec)
    state = AnomalyEpisodeStateV1()
    days: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "before_fires": 0, "after_fires": 0, "episodes": 0, "turn_explained_windows": 0,
        "before_keys": {}, "after_keys": {},
    })
    after_fires: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    for at, _, it in events:
        if it["k"] == "t":
            if it["phase"] == "start":
                tracker.observe(it["cid"], at)
            else:
                tracker.observe(it["cid"], at, closed=True, elapsed_sec=it.get("el"))
            continue
        d = days[_day(at)]
        if it.get("calm"):
            state = step(state, window_end=it["we"], window_end_iso="", flagged=False, ratio=it["r"],
                         lead_category=None, turn_active=False, calm_gap_sec=calm_gap_sec).state
            continue
        d["before_fires"] += 1
        key = f'{it["d"]}:{it["l"]}'
        weight = _dream_weight(it["r"])
        if weight:
            d["before_keys"][key] = max(d["before_keys"].get(key, 0.0), weight)
        decision = step(
            state,
            window_end=it["we"],
            window_end_iso=datetime.fromtimestamp(it["we"], tz=timezone.utc).isoformat(),
            flagged=True,
            ratio=it["r"],
            lead_category=it.get("c"),
            turn_active=tracker.active_between(it["ws"], it["we"]),
            calm_gap_sec=calm_gap_sec,
            turn_suppress=turn_suppress,
            sustained_windows=sustained_windows,
            sustained_horizon_sec=sustained_horizon_sec,
        )
        state = decision.state
        if state.episode_id and state.episode_id not in episode_ids:
            episode_ids.add(state.episode_id)
            d["episodes"] += 1
        if decision.turn_explained:
            d["turn_explained_windows"] += 1
        if decision.fire:
            state = mark_published(state, decision.severity)
            d["after_fires"] += 1
            if weight:
                d["after_keys"][key] = max(d["after_keys"].get(key, 0.0), weight)
            after_fires.append({"window_end": it["we"], "ratio": it["r"], "lead": it["l"],
                                "category": it.get("c"), "reason": decision.reason,
                                "severity": decision.severity})
    ordered = sorted(days)
    per_day = {}
    for i, day in enumerate(ordered):
        d = days[day]
        row = {k: v for k, v in d.items() if not k.endswith("_keys")}
        row["before_dream_keys"] = len(d["before_keys"])
        row["after_dream_keys"] = len(d["after_keys"])
        row["lost_dream_keys"] = sorted(set(d["before_keys"]) - set(d["after_keys"]))
        # Day-granular stand-in for orion-dream's 48h novelty lookback: weight of
        # this day's keys that did not appear on either of the two days before.
        for side in ("before", "after"):
            seen = set().union(*(days[x][f"{side}_keys"] for x in ordered[max(0, i - 2):i]))
            row[f"{side}_new_key_pressure"] = round(
                sum(w for k, w in d[f"{side}_keys"].items() if k not in seen), 3)
        per_day[day] = row
    return {"per_day": per_day, "after_fires": after_fires}


def fires_between(result: dict[str, Any], lo_iso: str, hi_iso: str) -> list[dict[str, Any]]:
    lo, hi = _ts(lo_iso), _ts(hi_iso)
    return [f for f in result["after_fires"] if lo <= f["window_end"] < hi]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-sql", action="store_true")
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--source", choices=("triggers", "scores"), default="triggers")
    ap.add_argument("--multiplier", type=float, default=3.0)
    ap.add_argument("--rows", help="psql JSON-lines output of --export-sql")
    ap.add_argument("--fixture", help="compact fixture (JSON lines) instead of --rows")
    ap.add_argument("--write-fixture", help="write the compact form of --rows here")
    ap.add_argument("--calm-gap-sec", type=float, default=DEFAULT_CALM_GAP_SEC)
    ap.add_argument("--turn-carry-sec", type=float, default=DEFAULT_TURN_CARRY_SEC)
    ap.add_argument("--no-turn-suppress", action="store_true")
    ap.add_argument("--sustained-windows", type=int, default=DEFAULT_SUSTAINED_WINDOWS)
    ap.add_argument("--sustained-horizon-sec", type=float, default=DEFAULT_SUSTAINED_HORIZON_SEC)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.export_sql:
        print(export_sql(args.days, args.source, args.multiplier))
        return 0
    if args.rows:
        items = compact(load_rows(Path(args.rows).read_text().splitlines()))
        if args.write_fixture:
            Path(args.write_fixture).write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in items))
    elif args.fixture:
        items = [json.loads(x) for x in Path(args.fixture).read_text().splitlines() if x.strip()]
    else:
        ap.error("need --export-sql, --rows or --fixture")
    result = replay(items, calm_gap_sec=args.calm_gap_sec, turn_carry_sec=args.turn_carry_sec,
                    sustained_horizon_sec=args.sustained_horizon_sec,
                    turn_suppress=not args.no_turn_suppress, sustained_windows=args.sustained_windows)
    if args.json:
        print(json.dumps(result["per_day"], indent=2))
        return 0
    print(f"{'day':<11} {'before':>6} {'after':>5} {'episodes':>8} {'turn_expl':>9} {'dream_keys b/a':>14} {'new_key_pressure b/a':>20}")
    tb = ta = 0
    for day, d in result["per_day"].items():
        tb += d["before_fires"]
        ta += d["after_fires"]
        print(f"{day:<11} {d['before_fires']:>6} {d['after_fires']:>5} {d['episodes']:>8} "
              f"{d['turn_explained_windows']:>9} {d['before_dream_keys']:>7}/{d['after_dream_keys']:<6} "
              f"{d['before_new_key_pressure']:>10}/{d['after_new_key_pressure']:<9}"
              + (f" lost={d['lost_dream_keys']}" if d["lost_dream_keys"] else ""))
    print(f"{'total':<11} {tb:>6} {ta:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
