"""Adaptive-range replay: would the surprise calm check pass with app/adaptive.py ranges? Read-only.

For each past day, rebuilds every stored champion forecast's 9 quantiles with ``adaptive_offsets``
(the shipped code), using only rows whose target was known before that origin, then runs the same
calm check as ``/series/{id}/surprise`` (``app.surprise.summarize``) on the rebuilt ranges and on
the ranges actually served. Prints, per series and horizon: days calm (target ~95 %), mean excess,
80 % coverage.

    cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_adaptive_interval_replay_eval.py --days 14

Baseline 2026-11-10, --days 14 (~6 min). Days calm, adaptive vs served ranges:
  cabinet 5 min 11/11 vs 2/13, 15 min 10/11 vs 7/13, 60 min 9/11 vs 8/13; cov80 0.80-0.82 vs 0.85-0.91
  office  5 min 5/5 vs 1/6,    15 min 4/5 vs 2/6,    60 min 5/5 vs 3/6;   cov80 0.79-0.80 vs 0.87-0.93
Horizons past 60 min keep the backtest range, so their "adaptive" lines are empty.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app.adaptive import POOL, adaptive_offsets  # noqa: E402
from app.backtest import QUANTILES, q_key  # noqa: E402
from app.series import load_registry  # noqa: E402
from app.store import PostgresStore  # noqa: E402
from app.surprise import summarize  # noqa: E402


def replay(rows: list[dict], horizons: list[int], grain_min: int) -> list[dict]:
    """Rebuilt champion rows (origin_ts, horizon_min, actual, quantiles) for every origin that has an
    adaptive range; the rest are dropped (they would have kept the backtest range)."""
    rows = sorted(rows, key=lambda r: r["origin_ts"])
    known_at = np.array([(pd.Timestamp(r["origin_ts"]) + pd.Timedelta(minutes=r["horizon_min"])).value for r in rows])
    order = np.argsort(known_at)
    known_sorted = known_at[order]
    out = []
    origins = sorted({pd.Timestamp(r["origin_ts"]) for r in rows})
    by_origin: dict[pd.Timestamp, list[dict]] = {}
    for r in rows:
        by_origin.setdefault(pd.Timestamp(r["origin_ts"]), []).append(r)
    for o in origins:
        lo = np.searchsorted(known_sorted, (o - POOL).value, "left")
        hi = np.searchsorted(known_sorted, o.value, "right")
        pool = [rows[i] for i in order[lo:hi]]
        offs, _ = adaptive_offsets(pool, o, horizons, grain_min)
        for r in by_origin[o]:
            hm = int(r["horizon_min"])
            if hm in offs:
                out.append({"origin_ts": r["origin_ts"], "horizon_min": hm, "actual": r["actual"],
                            "quantiles": {q_key(q): float(r["point"]) + float(offs[hm][k]) for k, q in enumerate(QUANTILES)}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--series", default=None, help="one series id (default: every target)")
    args = ap.parse_args()
    store = PostgresStore(os.environ["POSTGRES_URI"])
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    for spec in load_registry(SERVICE.parents[1] / "config" / "prediction" / "series.v1.yaml").targets():
        if args.series and spec.series_id != args.series:
            continue
        rows = [r for r in store._all(
            "SELECT origin_ts, horizon_min, point, actual, quantiles FROM prediction_forecast WHERE series_id = %s "
            "AND role = 'champion' AND score_status = 'scored' AND target_ts >= %s", (spec.series_id, since))]
        if not rows:
            continue
        rebuilt = replay(rows, list(spec.horizons_min), spec.grain_min)
        for name, rs in (("served", rows), ("adaptive", rebuilt)):
            days: dict = {}
            for r in rs:
                days.setdefault(pd.Timestamp(r["origin_ts"]).floor("D"), []).append(r)
            first = min(days) + POOL if days else None   # the adaptive pool needs POOL of history
            per_h: dict[str, list] = {}
            for day, drs in sorted(days.items()):
                if day < first:
                    continue
                for hm, h in summarize(drs, spec.grain_min).items():
                    q = [r for r in drs if int(r["horizon_min"]) == int(hm)]
                    cov = np.mean([r["quantiles"]["0.1"] <= r["actual"] <= r["quantiles"]["0.9"] for r in q])
                    per_h.setdefault(hm, []).append((h["calm"], h["mean_excess"], cov))
            for hm, v in sorted(per_h.items(), key=lambda kv: int(kv[0])):
                judged = [c for c, _, _ in v if c is not None]
                print(f"{spec.series_id:26s} {hm:>5s} min {name:8s} days calm {sum(judged)}/{len(judged)}  "
                      f"mean excess {np.mean([e for _, e, _ in v if e is not None]):+.3f}  "
                      f"cov80 {np.mean([c for _, _, c in v]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
