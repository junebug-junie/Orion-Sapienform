"""Family C Experiment 1, Patch 0 (live, read-only): is ~16 nights enough? And is each outcome real signal?

Design acceptance check: recompute from the last ``--days`` days the per-block spread of every outcome and the
minimum detectable effect (MDE) at the planned blocks per arm. Go only if the GPU3-watt MDE is <= 10 W.

Pseudo-blocks from history (NOT a randomised comparison; only the spread is used):
  * "session" blocks start when a town session opened (``grammar_acts`` actor ``aitown.session``);
  * "quiet" blocks start 40 min before a session opened, when no session overlapped [start, start + 35 min).
Each block's outcome is read over the experiment's window [start + 2, start + 35) min, exactly as the effect
table will (``family_c.lane_outcomes``; Family A readings minus their time-of-day profile). The spread is the
stratified residual SD (night x arm cells), the same one the estimator uses. MDE = 2.8 x SD x sqrt(2 / n).

Also the metric gate's live-sanity line for the lane-cost outcomes: share of blocks with no value (None is
never 0), the calm (quiet) level, and whether the calm level is near zero or floored.

Run (read-only): POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_town_hold_replay_eval.py --days 14
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3])]

from app import family_a as fa  # noqa: E402
from app.family_c import _profile_for, lane_outcomes  # noqa: E402

CONFIG = HERE.parents[3] / "config" / "prediction" / "family_a.v1.yaml"
READINGS = ("circe.gpu3_watts", "circe.gpu3_temp_c", "circe.chassis_watts", "athena.cabinet_temp_c")
LANES = ("lane.fast.tier1_hold_s", "lane.fast.grant_wait_s", "lane.fast.grant_wait_p95_s", "lane.kev.latency_ms_p50")
WIN0, WIN1, HOLD = 2, 35, 40
DEN = ZoneInfo("America/Denver")


def stratified_sd(blocks: list[dict], key: str) -> tuple[float | None, int, int]:
    cells: dict = {}
    for b in blocks:
        v = b.get(key)
        if v is None or not math.isfinite(v):
            continue
        cells.setdefault((b["night"], b["arm"]), []).append(v)
    ss = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in cells.values())
    n = sum(len(v) for v in cells.values())
    dof = n - len(cells)
    return (math.sqrt(ss / dof) if dof > 0 else None), n, len(cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--per-arm", type=int, default=56, help="planned blocks per arm (design: 16 nights x ~3.5)")
    ap.add_argument("--go-mde-w", type=float, default=10.0)
    args = ap.parse_args()
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        print("POSTGRES_URI is required (read-only)")
        return 2
    from app.family_a_load import load_frame
    from app.store import PostgresStore

    store = PostgresStore(uri)
    cfg = fa.parse_config(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    acts = store.town_acts(start, end)
    opens = sorted(pd.Timestamp(a["opened_at"]) for a in acts)
    spans = [(pd.Timestamp(a["opened_at"]), pd.Timestamp(a["closed_at"] or end)) for a in acts]
    blocks = []
    for o in opens:
        blocks.append({"arm": "session", "start": o})
        q = o - pd.Timedelta(minutes=HOLD)
        qa, qb = q + pd.Timedelta(minutes=WIN0), q + pd.Timedelta(minutes=WIN1)
        if not any(s0 < qb and qa < s1 for s0, s1 in spans):
            blocks.append({"arm": "quiet", "start": q})
    for b in blocks:
        b["night"] = b["start"].tz_convert(DEN).date().isoformat()
        b["w0"], b["w1"] = b["start"] + pd.Timedelta(minutes=WIN0), b["start"] + pd.Timedelta(minutes=WIN1)
    print(f"window {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC: {len(acts)} town sessions, "
          f"{sum(b['arm'] == 'session' for b in blocks)} session blocks, {sum(b['arm'] == 'quiet' for b in blocks)} quiet blocks")
    if not blocks:
        print("no town sessions in the window: nothing to replay")
        return 1

    inputs = [i for i in cfg.inputs if i.id in READINGS]
    f0 = (start - timedelta(days=14)).replace(minute=0)
    frame, _ = load_frame(uri, inputs, cfg.grain_min, f0, end)
    for rid in READINGS:
        if rid not in frame.columns:
            continue
        col = frame[rid].astype(float)
        hist = col[col.index < pd.Timestamp(start)]
        span = col[col.index >= pd.Timestamp(start)]
        prof = _profile_for(hist, span, cfg.grain_min)
        adj = span
        if prof is not None:
            tod = (span.index.hour * 60 + span.index.minute) // cfg.grain_min * cfg.grain_min
            adj = span - prof.reindex(tod).to_numpy()
        for b in blocks:
            v = adj[(adj.index >= b["w0"]) & (adj.index < b["w1"])].dropna()
            b[rid] = float(v.mean()) if len(v) else None

    # Lane outcomes: one read per local night of the window.
    for night in sorted({b["night"] for b in blocks}):
        bs = [b for b in blocks if b["night"] == night]
        a, z = min(b["w0"] for b in bs), max(b["w1"] for b in bs)
        leases = store.lane_leases(a.to_pydatetime(), z.to_pydatetime())
        kev = store.kev_latencies(a.to_pydatetime(), z.to_pydatetime())
        for b in bs:
            b.update(lane_outcomes(leases, kev, b["w0"], b["w1"]))

    print(f"\nper-block spread (stratified residual SD, night x arm) and MDE at {args.per_arm} blocks per arm "
          f"(80% power, two-sided 5%; MDE uses only the measured share of blocks):")
    print(f"{'outcome':32s} {'n':>4s} {'missing':>8s} {'SD':>9s} {'MDE':>9s} {'quiet mean':>11s} {'session mean':>13s}")
    go = None
    for key in (*READINGS, *LANES):
        sd, n, _cells = stratified_sd(blocks, key)
        miss = sum(1 for b in blocks if b.get(key) is None)
        q = [b[key] for b in blocks if b["arm"] == "quiet" and b.get(key) is not None]
        s = [b[key] for b in blocks if b["arm"] == "session" and b.get(key) is not None]
        # A block with no value (no tier-1 call in its window, say) does not count: the MDE uses the measured
        # share of the planned blocks, not all of them.
        share = 1 - miss / max(len(blocks), 1)
        n_arm = args.per_arm * share
        mde = 2.8 * sd * math.sqrt(2 / n_arm) if sd is not None and n_arm >= 2 else None
        print(f"{key:32s} {n:4d} {miss / max(len(blocks), 1):8.0%} "
              f"{sd if sd is not None else float('nan'):9.3f} {mde if mde is not None else float('nan'):9.3f} "
              f"{np.mean(q) if q else float('nan'):11.3f} {np.mean(s) if s else float('nan'):13.3f}")
        if key == "circe.gpu3_watts":
            go = mde is not None and mde <= args.go_mde_w
    print("\n(quiet/session means are history, not a randomised comparison; Family A readings are minus their "
          "time-of-day profile, so ~0 is 'usual for this hour')")
    print(f"\nPatch 0 verdict: {'GO' if go else 'NO-GO'} (GPU3-watt MDE <= {args.go_mde_w:g} W at {args.per_arm} per arm)")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
