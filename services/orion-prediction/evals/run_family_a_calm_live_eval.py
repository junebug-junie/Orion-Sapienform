"""Family A calm check on stored joint rows (prediction_joint_surprise), read-only.

The gate before any consumer (the next patch replaces node:substrate.biometrics with this score): over
>= 7 days of live rows issued by the running loop (app/family_a_live.py), the joint excess must rest at 0
and come back to it:

- mean excess, 2-hour block-bootstrap 95 % interval (blocks never cross a gap): contains 0;
- share of p <= 0.05 in 0.02..0.09 (calibrated: 0.05);
- share of hours at rest (hourly mean excess <= 0) within 0.10 of a calibrated score's share (~0.55);
- no stretch above +0.5 longer than 6 hours (a stuck-high score never comes back);
- per-reading 80 % range coverage (0.80 when calibrated) and the refusal rate, reported.

Served p, static p and adaptive p are reported side by side (both are stored on every row).

    cd services/orion-prediction
    POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_a_calm_live_eval.py --hours 168
    # dry run of the live loop itself, nothing written anywhere: replays the last N hours through
    # FamilyALive with an in-memory store, reading raw history read-only
    POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_a_calm_live_eval.py --replay-hours 24

Exit 0 = PASS, 1 = FAIL, 3 = NOT YET (fewer than --min-days of scored rows).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3])]

from app import family_a as fa  # noqa: E402
from app import family_a_live as fl  # noqa: E402

CONFIG = HERE.parents[3] / "config" / "prediction" / "family_a.v1.yaml"
REST_TOL = 0.10
P05_BAND = (0.02, 0.09)
MAX_HOURS_HIGH = 6


def _excess(p: np.ndarray, n_cal: np.ndarray) -> np.ndarray:
    return -np.log(p) - np.array([fa.calm_point(int(n)) for n in n_cal])


def summarize(rows: list[dict], grain_min: int, seed: int = 0) -> dict:
    """Per horizon: lifecycle counts and the calm statistics of served / static / adaptive p."""
    out: dict = {}
    by_h: dict[int, list[dict]] = {}
    for r in rows:
        by_h.setdefault(int(r["horizon_min"]), []).append(r)
    for h, rs in sorted(by_h.items()):
        status = Counter(r["status"] for r in rs)
        reasons = Counter(f"{k}:{','.join(v)}" for r in rs if r["status"] == "refused"
                          for k, v in (r.get("refusal") or {}).items())
        sc = [r for r in rs if r["status"] == "scored"]
        h_out: dict = {"status": dict(status), "refusal_reasons": dict(reasons.most_common(8)),
                       "calibration": dict(Counter(r.get("calibration") for r in sc))}
        if sc:
            idx = pd.DatetimeIndex([pd.Timestamp(r["target_ts"]) for r in sc])
            span_days = (idx.max() - idx.min()).total_seconds() / 86400.0 + grain_min / 1440.0
            h_out["days_scored"] = round(span_days, 2)
            served = np.array([float(r["excess"]) for r in sc])
            p_served = np.array([float(r["p"]) for r in sc])
            stats = {"served": (served, p_served)}
            # static / adaptive recomputed from the stored p with the calm point of the served pool size
            # (the served row's calm point is from its own pool; both pools are the run's validation rows)
            for name in ("p_static", "p_adaptive"):
                p = np.array([np.nan if r.get(name) is None else float(r[name]) for r in sc])
                calm = np.array([float(r["calm_point"]) for r in sc])
                stats[name] = (-np.log(p) - calm, p)
            for name, (ex, p) in stats.items():
                ok = np.isfinite(ex)
                if not ok.any():
                    h_out[name] = {"n": 0}
                    continue
                rest = fa.rest_summary(ex[ok], idx[ok], seed, grain_min=grain_min)
                rest["share_p_le_0_05"] = float(np.mean(p[ok] <= 0.05))
                rest["share_p_le_0_2"] = float(np.mean(p[ok] <= 0.2))
                h_out[name] = rest
        out[str(h)] = h_out
    return out


def verdict(summary: dict, min_days: float) -> tuple[str, list[str]]:
    why, ready = [], True
    for h, s in summary.items():
        sv = s.get("served") or {}
        if not sv.get("n"):
            return "NOT YET", [f"{h} min: no scored rows"]
        if s.get("days_scored", 0) < min_days:
            ready = False
            why.append(f"{h} min: {s['days_scored']} days scored < {min_days}")
        lo, hi = sv["mean_ci95"]
        if not (np.isfinite(lo) and lo <= 0 <= hi):
            why.append(f"{h} min: mean excess {sv['mean_excess']:+.3f} CI [{lo:+.3f}, {hi:+.3f}] excludes 0")
        if not P05_BAND[0] <= sv["share_p_le_0_05"] <= P05_BAND[1]:
            why.append(f"{h} min: p<=0.05 share {sv['share_p_le_0_05']:.3f} outside {P05_BAND}")
        if abs(sv["share_hours_at_rest"] - sv["null_share_hours_at_rest"]) > REST_TOL:
            why.append(f"{h} min: hours at rest {sv['share_hours_at_rest']:.2f} vs calibrated "
                       f"{sv['null_share_hours_at_rest']:.2f}")
        if sv["longest_hours_above_0_5"] > MAX_HOURS_HIGH:
            why.append(f"{h} min: {sv['longest_hours_above_0_5']} h in a row above +0.5")
    failed = [w for w in why if "days scored" not in w]
    if failed:
        return ("FAIL" if ready else "NOT YET (and failing so far)"), why
    return ("PASS" if ready else "NOT YET"), why


def coverage(readings: list[dict]) -> dict:
    """Per reading: live 80 % / 95 % coverage of the stored ranges."""
    by: dict[str, list] = {}
    for r in readings:
        if r.get("score_status") == "scored" and r.get("in_80") is not None:
            by.setdefault(r["series_id"][len(fl.PREFIX):], []).append((bool(r["in_80"]), bool(r["in_95"])))
    return {k: {"n": len(v), "cov80": float(np.mean([a for a, _ in v])), "cov95": float(np.mean([b for _, b in v]))}
            for k, v in sorted(by.items())}


def print_report(summary: dict, cov: dict, min_days: float) -> int:
    for h, s in summary.items():
        print(f"\n## {h} min ahead\n")
        print(f"rows: {s['status']}; served calibration: {s['calibration']}; days scored: {s.get('days_scored')}")
        if s["refusal_reasons"]:
            print(f"refusal reasons (top): {s['refusal_reasons']}")
        print("\n| p | n | mean excess | 95% CI | p<=0.05 | p<=0.2 | hours at rest (calibrated) | hourly min..max | "
              "longest h >0.5 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name in ("served", "p_static", "p_adaptive"):
            r = s.get(name) or {}
            if not r.get("n"):
                print(f"| {name} | 0 | | | | | | | |")
                continue
            print(f"| {name} | {r['n']} | {r['mean_excess']:+.3f} | [{r['mean_ci95'][0]:+.3f}, {r['mean_ci95'][1]:+.3f}] | "
                  f"{r['share_p_le_0_05']:.3f} | {r['share_p_le_0_2']:.3f} | {r['share_hours_at_rest']:.2f} "
                  f"({r['null_share_hours_at_rest']:.2f}) | {r['hourly_min']:+.2f}..{r['hourly_max']:+.2f} | "
                  f"{r['longest_hours_above_0_5']} |")
    if cov:
        c80 = [v["cov80"] for v in cov.values()]
        worst = sorted(cov.items(), key=lambda kv: abs(kv[1]["cov80"] - 0.8), reverse=True)[:5]
        print(f"\nPer-reading 80 % coverage: median {np.median(c80):.3f} over {len(cov)} readings; furthest from 0.80: "
              + ", ".join(f"{k} {v['cov80']:.2f} (n={v['n']})" for k, v in worst))
    v, why = verdict(summary, min_days)
    print(f"\nVerdict: {v}")
    for w in why:
        print(f"- {w}")
    return 0 if v == "PASS" else (1 if v == "FAIL" else 3)


def replay(uri: str, hours: float, select_every_h: float) -> tuple[list[dict], list[dict], dict]:
    """Run FamilyALive over the last ``hours`` with an in-memory store: select at the start (14 days
    before it), then tick every 5 min (score -> issue -> select when due), exactly as live. Read-only."""
    from app.family_a_load import load_frame

    cfg = fa.parse_config(yaml.safe_load(CONFIG.read_text()))
    g = cfg.grain_min
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    end = fl.floor_ts(now, g)
    t_start = end - timedelta(hours=hours)
    start = t_start - timedelta(days=fl.TRAIN_DAYS, hours=1)
    t0 = time.time()
    frame, _ = load_frame(uri, cfg.inputs, g, start, end)
    print(f"loaded {frame.shape} in {time.time() - t0:.0f}s", flush=True)
    store = fl.MemoryStore()
    live = fl.FamilyALive(store, cfg, fl.frame_loader(frame, g), settle_sec=60,
                          select_interval_sec=select_every_h * 3600)
    clock = t_start + timedelta(minutes=1, seconds=30)
    ticks = 0
    sel_seconds = []
    while clock < end:
        t1 = time.time()
        out = live.tick(clock)
        if "select" in out:
            sel_seconds.append(round(time.time() - t1, 1))
            print(f"{clock:%m-%d %H:%M} select rung={out['select']['rung']} targets={out['select']['n_targets']} "
                  f"({sel_seconds[-1]} s)", flush=True)
        if live.state.errors:
            print(f"{clock:%m-%d %H:%M} errors {live.state.errors}", flush=True)
        clock += timedelta(minutes=g)
        ticks += 1
    info = {"ticks": ticks, "select_seconds": sel_seconds, "counts": live.state.counts}
    print(f"replayed {ticks} ticks; counts {live.state.counts}")
    return store.joints, store.forecasts, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0, help="stored rows: look back this far")
    ap.add_argument("--min-days", type=float, default=7.0)
    ap.add_argument("--replay-hours", type=float, default=None,
                    help="dry run: replay the live loop over the last N hours in memory instead of reading stored rows")
    ap.add_argument("--select-every-hours", type=float, default=6.0, help="replay only")
    args = ap.parse_args()
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        ap.error("POSTGRES_URI is required")
    cfg = fa.parse_config(yaml.safe_load(CONFIG.read_text()))
    if args.replay_hours:
        joints, readings, _info = replay(uri, args.replay_hours, args.select_every_hours)
        print(f"\n# Replay dry run, last {args.replay_hours:g} h (in memory; nothing stored)")
    else:
        import psycopg
        from psycopg.rows import dict_row

        since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
        with psycopg.connect(uri, row_factory=dict_row) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            joints = conn.execute("SELECT * FROM prediction_joint_surprise WHERE family = %s AND target_ts >= %s "
                                  "ORDER BY horizon_min, target_ts", (fl.FAMILY, since)).fetchall()
            readings = conn.execute(
                "SELECT series_id, score_status, in_80, in_95 FROM prediction_forecast "
                "WHERE left(series_id, %s) = %s AND target_ts >= %s AND scored_at IS NOT NULL",
                (len(fl.PREFIX), fl.PREFIX, since)).fetchall()
        print(f"# Stored rows, last {args.hours:g} h: {len(joints)} joint rows, {len(readings)} scored reading rows")
    return print_report(summarize(joints, cfg.grain_min), coverage(readings), args.min_days)


if __name__ == "__main__":
    raise SystemExit(main())
