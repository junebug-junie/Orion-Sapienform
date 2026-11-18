"""Replay the surprise event gate + node aggregation over live scored forecasts, read-only.

What Orion's node:prediction.room / node:prediction.body WOULD have read over the window, had the
producer (app/surprise.py event_gate) and the substrate-runtime fold
(orion/substrate/prediction_surprise.py) been live. Prints, per node: events that pass the gate,
node writes, the signed mean-excess distribution, the stored (clipped 0..1) value distribution, and
how often it sits at exactly 0 (rest reachable?). Nothing is written.

    cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_surprise_node_replay_eval.py --hours 24

The node is written at each event's scored time (target bucket end + settle); the replay uses
``scored_at`` as the write clock.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app.store import PostgresStore  # noqa: E402
from app.surprise import CALM_SURPRISAL, conformal_p_value, event_gate, excess  # noqa: E402

def _load_fold():
    """orion/substrate/prediction_surprise.py by path: the orion.substrate package __init__ pulls in
    the graph store (requests, redis), which this image does not ship. The module itself is stdlib."""
    import importlib.util

    path = SERVICE.parents[1] / "orion" / "substrate" / "prediction_surprise.py"
    spec = importlib.util.spec_from_file_location("prediction_surprise", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_fold = _load_fold()
SERIES_NODE_IDS, PredictionSurpriseState = _fold.SERIES_NODE_IDS, _fold.PredictionSurpriseState

SQL = (
    "SELECT f.forecast_id, f.series_id, f.role, f.horizon_min, f.origin_ts, f.target_ts, f.scored_at, "
    "f.score_status, f.actual, f.quantiles, f.interval_source, "
    "r.assignment ->> f.horizon_min::text AS served_model "
    "FROM prediction_forecast f LEFT JOIN prediction_model_run r ON r.run_id = f.run_id "
    "WHERE f.series_id = ANY(%s) AND f.scored_at >= %s ORDER BY f.scored_at, f.forecast_id"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    store = PostgresStore(os.environ["POSTGRES_URI"])
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    rows = store._all(SQL, (list(SERIES_NODE_IDS), since))
    state = PredictionSurpriseState()
    refused: Counter = Counter()
    verdicts: Counter = Counter()
    by_node: dict[str, list[tuple[float, float, float, int]]] = {n: [] for n in SERIES_NODE_IDS.values()}
    singles: dict[str, list[float]] = {n: [] for n in SERIES_NODE_IDS.values()}
    for r in rows:
        score = {"score_status": r["score_status"], "actual": r["actual"]}
        ok, reason = event_gate(r, score)
        if not ok:
            refused[(r["series_id"], reason)] += 1
            continue
        p, _ = conformal_p_value(float(r["actual"]), r["quantiles"])
        event = {"series_id": r["series_id"], "horizon_min": r["horizon_min"], "origin_ts": r["origin_ts"],
                 "target_ts": r["target_ts"], "excess": excess(p), "forecast_id": str(r["forecast_id"]),
                 "calm_surprisal": CALM_SURPRISAL}
        now = r["scored_at"]
        node_id, verdict = state.add(event, now=now)
        verdicts[(r["series_id"], int(r["horizon_min"]), verdict)] += 1
        if verdict == "accepted":
            singles[node_id].append(event["excess"])
            reading = state.reading(node_id, now=now)
            by_node[node_id].append((reading.mean_excess, reading.z, reading.value, reading.n))
    print(f"window {args.hours:g} h, {len(rows)} scored rows")
    for (sid, reason), n in sorted(refused.items()):
        print(f"  gate refused {sid:24s} {reason:20s} {n}")
    for (sid, hm, verdict), n in sorted(verdicts.items()):
        print(f"  event        {sid:24s} {hm:>3d} min {verdict:20s} {n}")
    for node_id, writes in by_node.items():
        if not writes:
            print(f"{node_id}: no writes")
            continue
        mean = np.array([w[0] for w in writes])
        z = np.array([w[1] for w in writes])
        val = np.array([w[2] for w in writes])
        n = np.array([w[3] for w in writes])
        one = np.array(singles[node_id])
        q = lambda a, x: float(np.quantile(a, x))  # noqa: E731
        print(f"{node_id}: writes {len(writes)}  samples/window median {int(np.median(n))}  "
              f"single-outcome excess sd {one.std(ddof=1):.3f} (calm theory 0.835)")
        print(f"  mean_excess  mean {mean.mean():+.3f}  p10 {q(mean, .1):+.3f}  p50 {q(mean, .5):+.3f}  "
              f"p90 {q(mean, .9):+.3f}  max {mean.max():+.3f}")
        print(f"  z            mean {z.mean():+.3f}  sd {z.std(ddof=1):.3f}  p90 {q(z, .9):+.3f}  max {z.max():+.3f}")
        print(f"  node value   exact 0: {np.mean(val == 0.0):.0%}  p50 {q(val, .5):.3f}  p90 {q(val, .9):.3f}  "
              f"p99 {q(val, .99):.3f}  max {val.max():.3f}  share >= 0.55 (curiosity bar): "
              f"{np.mean(val >= 0.55):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
