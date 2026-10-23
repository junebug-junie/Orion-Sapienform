"""Camera surprise: what the tick publishes, replayed from per-frame baselines.

Metric gate record for fix/camera-surprise-honest-input (2026-10-21,
docs/superpowers/pr-reports/2026-10-21-camera-surprise-honest-input-pr.md).

Every embedding the scorer sees writes one row to
``substrate_perception_embedding_baseline`` with the stream's surprise EWMA.
The per-frame score is recoverable exactly from consecutive rows (checked
against the live log, see evals/test_camera_surprise_input_eval.py):

    raw   = (m - (1 - a) * m_prev) / a            a = 0.2
    z     = max(0, (raw - m_prev) / sqrt(max(v_prev, 2e-6)))
    score = min(1, z / 3)

Each tick (the stored tick times in substrate_node_prediction_error_history)
is then rebuilt two ways:

- ``latest``: the latest scored frame since the previous tick (what ships);
- ``peak``: the max scored frame since the previous tick (rejected);

and a tick with no frame in the last 60 s, or no new scored frame, is MISSING
(no history row), not 0.0. For each it reports the rest point (share of exact
0.0), p50/p90/p99, share pinned at the 1.0 ceiling, and the share at or above
endogenous curiosity's fixed ``min_prediction_error`` 0.55. It also replays
the ladder watch's camera_input rung (cron every 10 min) over the frame times.

Read-only (SELECT only). Exit 1 when the shipped mode fails the gate: rest
share under 30%, or ceiling share over 5%.

Run:
  POSTGRES_URI=postgresql://... python services/orion-substrate-runtime/evals/run_camera_surprise_input_eval.py --days 7
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALPHA = 0.2
MIN_VARIANCE = 2e-6
SATURATION = 3.0
STALE_SEC = 60.0
CURIOSITY_MIN_ERROR = 0.55
STREAM_NODE = {"cam0": "node:substrate.perception", "walkway": "node:substrate.perception_walkway"}
GATE_MIN_REST_SHARE = 0.30
GATE_MAX_CEILING_SHARE = 0.05

Frame = tuple[datetime, float | None]


def frame_scores(rows: Iterable[tuple[datetime, float, float, int]]) -> list[Frame]:
    """(time, score|None) per stored baseline row (time, surprise ewma,
    surprise variance, surprise n). None where no z-score existed yet or a row
    is missing between two (surprise n did not advance by exactly one)."""
    out: list[Frame] = []
    prev: tuple[float, float, int] | None = None
    for t, m, v, n in rows:
        score = None
        if prev is not None and n == prev[2] + 1 and prev[2] >= 1:
            raw = (m - (1.0 - ALPHA) * prev[0]) / ALPHA
            z = max(0.0, (raw - prev[0]) / math.sqrt(max(prev[1], MIN_VARIANCE)))
            score = min(1.0, z / SATURATION)
        out.append((t, score))
        prev = (m, v, n)
    return out


def tick_series(
    ticks: Sequence[datetime], frames: Sequence[Frame], *, mode: str = "latest"
) -> tuple[list[tuple[datetime, float]], dict[str, int]]:
    """Rebuild each tick's published value; missing ticks get no row."""
    times = [f[0] for f in frames]
    rows: list[tuple[datetime, float]] = []
    missing = {"stale": 0, "no_scored_frame": 0}
    prev: datetime | None = None
    for t in ticks:
        lo = bisect.bisect_right(times, prev) if prev is not None else bisect.bisect_left(times, t - timedelta(seconds=30))
        hi = bisect.bisect_right(times, t)
        prev = t
        if hi == 0 or (t - times[hi - 1]).total_seconds() >= STALE_SEC:
            missing["stale"] += 1
            continue
        scores = [s for _, s in frames[lo:hi] if s is not None]
        if not scores:
            missing["no_scored_frame"] += 1
            continue
        rows.append((t, max(scores) if mode == "peak" else scores[-1]))
    return rows, missing


def _q(sorted_vals: Sequence[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def distribution(values: Sequence[float]) -> dict:
    s = sorted(values)
    n = len(s)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "rest_share": round(sum(v == 0.0 for v in s) / n, 4),
        "p50": _q(s, 0.5), "p90": _q(s, 0.9), "p99": _q(s, 0.99),
        "ceiling_share": round(sum(v >= 1.0 for v in s) / n, 4),
        "curiosity_share": round(sum(v >= CURIOSITY_MIN_ERROR for v in s) / n, 4),
    }


def gate(dist: dict) -> dict[str, bool]:
    return {
        "can_rest": dist.get("rest_share", 0.0) >= GATE_MIN_REST_SHARE,
        "not_ceiling_bound": dist.get("ceiling_share", 1.0) <= GATE_MAX_CEILING_SHARE,
    }


def alarm_replay(
    frame_times: Sequence[datetime], *, start: datetime, end: datetime,
    cron: timedelta = timedelta(minutes=10),
) -> list[tuple[str, datetime, datetime | None]]:
    """("red"|"green", cron time, newest frame) transitions of the camera_input rung."""
    from orion import substrate_ladder_liveness as ll

    rung = next(r for r in ll.RUNGS if r.name.startswith("camera_input:"))
    out: list[tuple[str, datetime, datetime | None]] = []
    red = False
    t = start
    while t < end:
        i = bisect.bisect_right(frame_times, t)
        newest = frame_times[i - 1] if i else None
        r = ll.evaluate_rung(rung, newest, t)
        if r.red != red:
            out.append(("red" if r.red else "green", t, newest))
            red = r.red
        t += cron
    return out


def evaluate(baseline_rows: dict[str, list], ticks: dict[str, list[datetime]]) -> dict:
    report: dict = {"streams": {}, "ok": True}
    for stream, rows in baseline_rows.items():
        node = STREAM_NODE[stream]
        frames = frame_scores(rows)
        entry: dict = {"node": node, "frames": distribution([s for _, s in frames if s is not None])}
        for mode in ("latest", "peak"):
            series, missing = tick_series(ticks.get(node, []), frames, mode=mode)
            dist = distribution([v for _, v in series])
            entry[mode] = {**dist, "missing": missing, "gate": gate(dist)}
        report["ok"] = report["ok"] and all(entry["latest"]["gate"].values())
        if frames:
            entry["alarms"] = [
                (state, t.isoformat(), newest.isoformat() if newest else None)
                for state, t, newest in alarm_replay(
                    [f[0] for f in frames], start=frames[0][0], end=frames[-1][0]
                )
            ]
        report["streams"][stream] = entry
    return report


def _fetch(dsn: str, days: float) -> tuple[dict, dict]:
    import psycopg2

    since = datetime.now(timezone.utc) - timedelta(days=days)
    baseline: dict[str, list] = {}
    ticks: dict[str, list[datetime]] = {}
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        for stream, node in STREAM_NODE.items():
            cur.execute(
                "SELECT generated_at, (baseline_json->'surprise'->>'ewma')::float8, "
                "(baseline_json->'surprise'->>'variance')::float8, (baseline_json->'surprise'->>'n')::int "
                "FROM substrate_perception_embedding_baseline WHERE stream_id = %s AND generated_at >= %s "
                "ORDER BY generated_at",
                (stream, since),
            )
            baseline[stream] = [(r[0], float(r[1]), float(r[2]), int(r[3])) for r in cur.fetchall()]
            cur.execute(
                "SELECT observed_at FROM substrate_node_prediction_error_history "
                "WHERE node_id = %s AND observed_at >= %s ORDER BY observed_at",
                (node, since),
            )
            ticks[node] = [r[0] for r in cur.fetchall()]
    return baseline, ticks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI", ""))
    ap.add_argument("--days", type=float, default=7.0)
    args = ap.parse_args(argv)
    if not args.dsn:
        print("POSTGRES_URI or --dsn required", file=sys.stderr)
        return 2
    baseline, ticks = _fetch(args.dsn, args.days)
    report = evaluate(baseline, ticks)
    for stream, e in report["streams"].items():
        print(f"{stream} -> {e['node']}")
        print(f"  frames : {e['frames']}")
        for mode in ("latest", "peak"):
            print(f"  {mode:6} : {e[mode]}")
        for state, t, newest in e.get("alarms", []):
            print(f"  alarm {state:5} at {t} (newest frame {newest})")
    print("GATE", "PASS" if report["ok"] else "FAIL", "(shipped mode: latest)")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
