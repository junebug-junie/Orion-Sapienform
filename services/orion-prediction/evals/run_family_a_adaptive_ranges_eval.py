"""Family A per-reading 80 % ranges: served (validation residuals, one width) vs adaptive (#3310), read-only.

For every stored, scored family A champion row in the window, rebuild the adaptive range the live loop
would have served (app/adaptive.adaptive_offsets on that reading's own scored rows from the POOL before
the row's origin, owner = the rung that served it, difficulty = its last hour's misses), then compare
80 % coverage and width with the range actually served. Rows where the adaptive method falls back keep
the served range, exactly as live would.

    POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_a_adaptive_ranges_eval.py [--days 2] [--stride 2]

Prints per reading: n, served cov80, adaptive cov80, adaptive share, median width ratio; then per day for
the busiest readings; then PASS when every reading with n >= 200 sits in [0.72, 0.88] adaptive (the
#3310 check), else FAIL with the list. Exit 0 always (a report).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]

from app.adaptive import MAX_HORIZON, POOL, adaptive_offsets  # noqa: E402
from app.backtest import QUANTILES  # noqa: E402
from app.family_a_live import PREFIX  # noqa: E402
from app.store import PostgresStore  # noqa: E402

BAND = (0.72, 0.88)
K10, K90 = QUANTILES.index(0.1), QUANTILES.index(0.9)


def evaluate(rows: list[dict], start: datetime, end: datetime, stride: int, grain_min: int = 5) -> dict:
    """{reading: list of (target_ts, served_in80, adaptive_in80, adaptive_used, width_ratio)}."""
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_t[r["series_id"]].append(r)
    out: dict[str, list] = {}
    for sid, rs in by_t.items():
        rs.sort(key=lambda r: r["origin_ts"])
        tgt = np.array([pd.Timestamp(r["target_ts"]).value for r in rs])
        origins = sorted({r["origin_ts"] for r in rs if start <= r["origin_ts"] < end})[::stride]
        res = []
        for o in origins:
            here = [r for r in rs if r["origin_ts"] == o]
            lo_i, hi_i = np.searchsorted(tgt, [pd.Timestamp(o - POOL).value, pd.Timestamp(o).value], side="right")
            pool = rs[lo_i:hi_i]                       # targets in (o - POOL, o]: known when o was issued
            owners = {int(r["horizon_min"]): r["owner"] for r in here}
            offs, _why = adaptive_offsets(pool, pd.Timestamp(o), sorted(owners), grain_min, owners=owners)
            for r in here:
                h, a, p = int(r["horizon_min"]), float(r["actual"]), float(r["point"])
                served = bool(r["in_80"])
                if h in offs:
                    lo, hi = p + offs[h][K10], p + offs[h][K90]
                    res.append((r["target_ts"], served, lo <= a <= hi, True, float(offs[h][K90] - offs[h][K10])))
                else:
                    res.append((r["target_ts"], served, served, False, float("nan")))
        out[sid[len(PREFIX):]] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=2, help="every n-th origin (speed); 1 = all")
    a = ap.parse_args()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=a.days)
    store = PostgresStore(os.environ["POSTGRES_URI"])
    rows = store.family_champion_pool(PREFIX, start - POOL, int(MAX_HORIZON.total_seconds() // 60))
    widths = {(r["series_id"], r["origin_ts"], int(r["horizon_min"])): r for r in rows}
    store.close()
    res = evaluate(rows, start, end, a.stride)
    print(f"window {start:%Y-%m-%d %H:%M}Z .. {end:%Y-%m-%d %H:%M}Z, {len(rows)} pool rows, stride {a.stride}")
    print(f"{'reading':32s} {'n':>5s} {'served':>7s} {'adapt':>7s} {'used':>5s}")
    bad = []
    for t in sorted(res):
        r = res[t]
        if not r:
            continue
        n = len(r)
        sv = np.mean([x[1] for x in r]); ad = np.mean([x[2] for x in r]); used = np.mean([x[3] for x in r])
        print(f"{t:32s} {n:5d} {sv:7.3f} {ad:7.3f} {used:5.2f}")
        if n >= 200 and not (BAND[0] <= ad <= BAND[1]):
            bad.append((t, round(float(ad), 3)))
    print("\nper day (busiest readings):")
    for t in ("athena.load_1m", "athena.cpu_watts", "athena.board_temp_c", "athena.chassis_watts"):
        if t not in res:
            continue
        by_day: dict = defaultdict(list)
        for x in res[t]:
            by_day[x[0].date()].append(x)
        for d, xs in sorted(by_day.items()):
            print(f"  {t:24s} {d}  n={len(xs):4d} served={np.mean([x[1] for x in xs]):.3f} "
                  f"adaptive={np.mean([x[2] for x in xs]):.3f}")
    print("\nPASS" if not bad else f"\nFAIL: outside {BAND}: {bad}")


if __name__ == "__main__":
    main()
