"""Read-only replay: what would world-first attention have attended to?

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md,
acceptance check 3 (estimated offline before the 48 h live check).

Replays the field contest's world-first ranking on a time grid over stored data:
- substrate_node_prediction_error_history (every node's readings),
- chat_history_log (Juniper's turns, the chat source),
- substrate_field_state (the vision organ's frame staleness, for camera absence),
and compares with what the stored frames (substrate_attention_frames) actually
crowned under the old ranking. It also re-checks real body spikes in the window
and injects a synthetic sustained storm to see whether the baseline absorbs it.

SELECT only. Usage:
    python services/orion-attention-runtime/evals/replay_world_first.py \
        --dsn postgresql://postgres:postgres@localhost:55432/conjourney --days 3
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import (  # noqa: E402
    CHAT_RATE_WINDOW,
    PERCEPTION_NODE_ID,
    chat_candidate,
    node_candidate,
    rank_candidates,
)
from orion.substrate.prediction_error_magnitude import (  # noqa: E402
    WINDOW_7D,
    compute_prediction_error_magnitude,
)

CALM_RAW = 0.05


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


@dataclass
class NodeSeries:
    times: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def at(self, t: datetime) -> tuple[float, datetime, list[tuple[datetime, float]]] | None:
        hi = bisect.bisect_right(self.times, t)
        if hi == 0:
            return None
        lo = bisect.bisect_left(self.times, t - WINDOW_7D)
        hist = list(zip(self.times[lo:hi], self.values[lo:hi]))
        return self.values[hi - 1], self.times[hi - 1], hist


def build_series(pe_rows) -> dict[str, NodeSeries]:
    series: dict[str, NodeSeries] = {}
    for node_id, observed_at, value in sorted(pe_rows, key=lambda r: (r[0], _aware(r[1]))):
        s = series.setdefault(node_id, NodeSeries())
        s.times.append(_aware(observed_at))
        s.values.append(float(value))
    return series


def staleness_at(points: list[tuple[datetime, float | None]], t: datetime) -> float | None:
    times = [p[0] for p in points]
    i = bisect.bisect_right(times, t)
    return points[i - 1][1] if i else None


def tick(series, turns, staleness_points, t):
    cands = []
    for node_id, s in series.items():
        got = s.at(t)
        if got is None:
            continue
        value, observed_at, hist = got
        mag = compute_prediction_error_magnitude(value=value, observed_at=observed_at, history=hist, now=t)
        absent = None
        if node_id == PERCEPTION_NODE_ID and staleness_points is not None:
            st = staleness_at(staleness_points, t)
            if st is None:
                absent = "camera health unmeasured"
            elif st >= 1.0:
                absent = "camera frames stale"
        cands.append(node_candidate(node_id=node_id, label=node_id, magnitude=mag,
                                    observed_at=observed_at, now=t, absent_reason=absent))
    cands.append(chat_candidate([x for x in turns if x <= t], now=t))
    return rank_candidates(cands)


def replay(pe_rows, turns, staleness_points, *, start: datetime, end: datetime, step: timedelta) -> dict:
    series = build_series(pe_rows)
    turns = sorted(_aware(t) for t in turns)
    n = 0
    no_winner = 0
    internal_calm = 0
    winners: Counter = Counter()
    kinds: Counter = Counter()
    t = start
    while t <= end:
        r = tick(series, turns, staleness_points, t)
        n += 1
        if r.winner is None:
            no_winner += 1
        else:
            c = r.winner.candidate
            winners[c.source_id] += 1
            kinds[c.source_kind] += 1
            if c.source_kind == "internal" and c.unusualness.value < CALM_RAW:
                internal_calm += 1
        t += step
    return {
        "ticks": n,
        "no_winner_share": round(no_winner / n, 4) if n else None,
        "internal_winner_raw_lt_0_05_share": round(internal_calm / n, 4) if n else None,
        "winner_share_by_kind": {k: round(v / n, 4) for k, v in kinds.items()},
        "winner_share_by_source": {k: round(v / n, 4) for k, v in winners.most_common()},
    }


def spike_check(pe_rows, turns, staleness_points, *, start, end, node_id, min_value) -> dict:
    """Every real reading of `node_id` >= min_value in the window: does it win
    on the tick right after it lands?"""
    series = build_series(pe_rows)
    turns = sorted(_aware(t) for t in turns)
    s = series.get(node_id) or NodeSeries()
    out = []
    for ts, v in zip(s.times, s.values):
        if not (start <= ts <= end) or v < min_value:
            continue
        r = tick(series, turns, staleness_points, ts + timedelta(seconds=30))
        w = r.winner.candidate.source_id if r.winner else None
        verdict = next((x for x in (*r.eligible, *r.ineligible) if x.candidate.source_id == node_id), None)
        out.append({"at": ts.isoformat(), "value": v, "winner": w,
                    "band": verdict.band if verdict else None,
                    "percentile": verdict.candidate.unusualness.percentile_now if verdict else None})
    won = sum(1 for o in out if o["winner"] == node_id)
    return {"node_id": node_id, "min_value": min_value, "readings": len(out), "won": won, "examples": out[:12]}


def storm_check(pe_rows, turns, staleness_points, *, node_id, storm_start, hours=5.0,
                every=timedelta(seconds=135), value=1.0, step=timedelta(minutes=5)) -> dict:
    """Inject a sustained storm (readings of `value` every `every` for `hours`)
    and report how much of it the body node still wins as its own baseline
    absorbs the storm -- the "real alarm suppressed" failure mode."""
    end = storm_start + timedelta(hours=hours)
    # The storm REPLACES the node's real readings in its window: during a real
    # storm the node reports storm values, not interleaved calm ones.
    injected = [
        r for r in pe_rows
        if not (r[0] == node_id and storm_start <= _aware(r[1]) <= end)
    ]
    t = storm_start
    while t <= end:
        injected.append((node_id, t, value))
        t += every
    series = build_series(injected)
    turns = sorted(_aware(x) for x in turns)
    wins = 0
    n = 0
    bands: Counter = Counter()
    t = storm_start + timedelta(seconds=30)
    while t <= end:
        r = tick(series, turns, staleness_points, t)
        n += 1
        wins += int(r.winner is not None and r.winner.candidate.source_id == node_id)
        v = next((x for x in (*r.eligible, *r.ineligible) if x.candidate.source_id == node_id), None)
        bands[v.band if v else "missing"] += 1
        t += step
    return {"node_id": node_id, "hours": hours, "ticks": n, "won_share": round(wins / n, 4) if n else None,
            "bands": dict(bands)}


def _load(dsn: str, since: datetime):
    from sqlalchemy import create_engine, text

    from orion.attention.world_first import CHAT_TURN_TIMES_SQL

    e = create_engine(dsn)
    with e.connect() as c:
        pe = c.execute(text(
            "SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history "
            "WHERE observed_at >= :s"), {"s": since}).fetchall()
        turns = [r[0] for r in c.execute(text(CHAT_TURN_TIMES_SQL), {"since": since.replace(tzinfo=None)})]
        st = c.execute(text(
            "SELECT generated_at, (field_json->'node_vectors'->'node:substrate.vision_organ'->>'vision_frame_staleness')::float "
            "FROM substrate_field_state WHERE generated_at >= :s ORDER BY generated_at"), {"s": since}).fetchall()
        old = c.execute(text(
            "SELECT count(*), "
            "sum(((frame_json->'dominant_targets'->0->'dominant_channels'->>'prediction_error')::float < 0.05)::int), "
            "sum((jsonb_array_length(frame_json->'dominant_targets') = 0)::int) "
            "FROM substrate_attention_frames WHERE generated_at >= :s"), {"s": since + WINDOW_7D}).one()
        old_w = c.execute(text(
            "SELECT frame_json->'dominant_targets'->0->>'target_id', count(*) FROM substrate_attention_frames "
            "WHERE generated_at >= :s GROUP BY 1 ORDER BY 2 DESC"), {"s": since + WINDOW_7D}).fetchall()
    return pe, turns, [(_aware(a), b) for a, b in st], old, old_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:55432/conjourney"))
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--step-sec", type=float, default=60.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=a.days)
    pe, turns, st, old, old_w = _load(a.dsn, start - WINDOW_7D)
    turns = [_aware(t) for t in turns]
    total, calm, empty = old
    report = {
        "window": [start.isoformat(), end.isoformat()],
        "note": "PE history starts 2026-10-03; early ticks see <7 days of history",
        "old_ranking_stored_frames": {
            "frames": total,
            "internal_winner_raw_lt_0_05_share": round((calm or 0) / total, 4) if total else None,
            "no_winner_share": round((empty or 0) / total, 4) if total else None,
            "winner_share_by_source": {str(k): round(v / total, 4) for k, v in old_w},
        },
        "world_first_replay": replay(pe, turns, st, start=start, end=end, step=timedelta(seconds=a.step_sec)),
        "body_spikes": [
            spike_check(pe, turns, st, start=start, end=end, node_id="node:substrate.execution", min_value=0.9),
            spike_check(pe, turns, st, start=start, end=end, node_id="node:substrate.biometrics", min_value=0.25),
        ],
        "synthetic_storm": storm_check(pe, turns, st, node_id="node:substrate.execution",
                                       storm_start=end - timedelta(hours=5)),
    }
    text_out = json.dumps(report, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text_out + "\n")
    print(text_out)


if __name__ == "__main__":
    main()
