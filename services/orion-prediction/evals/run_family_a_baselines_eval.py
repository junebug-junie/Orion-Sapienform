"""Family A slice 1 eval: hygiene gate, baseline rungs vs multivariate ridge, joint surprise. Read-only.

Loads every input in config/prediction/family_a.v1.yaml as 5-minute bucket means from the live raw
tables, screens them (app/family_a.py), then at 5 and 15 min ahead:

- backtests persistence, per-series AR and multivariate ridge on the repo's blocked purged split, one
  run per seed (seeds move the block edges);
- tests ridge against the better baseline per target on the per-origin joint loss: overlap-aware
  paired t AND one-day block bootstrap, both p < 0.05 for a win;
- computes the joint (whitened, conformal) surprise on held-out origins, and on the most recent
  ``--live-hours`` with a model trained only on the data before them (the live rest check), with the
  static and the adaptive calibration side by side (app/family_a.py ``ADAPTIVE``).

Usage (from services/orion-prediction):
    POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_a_baselines_eval.py --days 14 --seeds 0 1 \
        --out /mnt/telemetry/models/prediction_family_a/<stamp>

Writes nothing to the database. ``--out`` gets manifest.json (named inputs, train-only standardisation)
and report.json. Exit code 0 = ran; 2 = the hygiene gate refused (the refusal is the result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3])]

from app import family_a as fa  # noqa: E402
from app.family_a_load import load_frame  # noqa: E402

CONFIG = HERE.parents[3] / "config" / "prediction" / "family_a.v1.yaml"
NEURAL_MIN_DAYS = 21
# Readings whose univariate surprise already reaches the substrate (node:prediction.body / .room).
ALREADY_SCORED = ("athena.cabinet_temp_c", "office.temp_c")


def _floor(ts: datetime, grain: int) -> datetime:
    return ts.replace(second=0, microsecond=0, minute=ts.minute - ts.minute % grain)


def _p_shares(p: np.ndarray) -> dict:
    ok = np.isfinite(p)
    if not ok.any():
        return {"share_p_le_0_05": float("nan"), "share_p_le_0_2": float("nan")}
    return {"share_p_le_0_05": float(np.mean(p[ok] <= 0.05)), "share_p_le_0_2": float(np.mean(p[ok] <= 0.2))}


def common_days(frame: pd.DataFrame, cols: list[str]) -> int:
    """Whole days on which every given series holds >= 50 % of its buckets."""
    if not cols:
        return 0
    ok = frame[cols].notna().groupby(frame.index.floor("1D")).mean()
    return int((ok >= 0.5).all(axis=1).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--end", default=None, help="ISO end (default: now, floored to the grain)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--live-hours", type=int, default=24)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if len(args.seeds) < 2:
        ap.error("at least 2 seeds (spec)")
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        ap.error("POSTGRES_URI is required")

    cfg = fa.parse_config(yaml.safe_load(Path(args.config).read_text()))
    g = cfg.grain_min
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")) if args.end else datetime.now(timezone.utc)
    end = _floor(end.astimezone(timezone.utc), g)
    start = _floor(end - timedelta(days=args.days), g)   # a fractional --days must stay on the bucket grid
    t0 = time.time()
    frame, last_seen = load_frame(uri, cfg.inputs, g, start, end)
    print(f"loaded {frame.shape[0]} buckets x {frame.shape[1]} inputs {start:%Y-%m-%d %H:%M}..{end:%Y-%m-%d %H:%M} "
          f"UTC in {time.time() - t0:.1f}s", flush=True)

    live_rows = args.live_hours * 60 // g
    max_k = max(cfg.horizons_min) // g
    train_end_i = len(frame) - live_rows - fa.purge_rows(max_k)
    train_frame = frame.iloc[:train_end_i]

    # 1. Hygiene: screen on the training part only; the live window is gated separately below.
    verdicts = fa.screen_frame(train_frame, cfg.hygiene)
    targets = [k for k, v in verdicts.items() if v.status == fa.TARGET]
    covs = [k for k, v in verdicts.items() if v.status == fa.COVARIATE]
    inputs = targets + covs
    print("\n## Hygiene (training part)\n")
    print("| input | status | reasons | present | distinct | mode share | rest days | drift rho |")
    print("|---|---|---|---|---|---|---|---|")
    for iid, v in verdicts.items():
        s = v.stats
        print(f"| {iid} | {v.status} | {','.join(v.reasons)} | {s.get('present_share')} | {s.get('distinct', '')} | "
              f"{s.get('mode_share', '')} | {s.get('rest_day_share', '')} | {s.get('drift_rho', '')} |")
    named = {k: verdicts[k].status for k in inputs}
    report: dict = {"window": [start.isoformat(), end.isoformat()], "seeds": args.seeds,
                    "verdicts": {k: {"status": v.status, "reasons": list(v.reasons), "stats": v.stats}
                                 for k, v in verdicts.items()},
                    "targets": targets, "covariates": covs}
    try:
        fa.gate_named_inputs(named, train_frame, cfg.hygiene)
    except fa.HygieneRefused as exc:   # cannot happen by construction; kept as the guard it is
        print(f"REFUSED (train): {exc}")
        return 2
    if not targets:
        print("REFUSED: no input passed as a target")
        return 2

    # 2. Backtest per horizon x seed.
    print("\n## Backtest (held-out origins, standardised squared error, joint = mean over targets)\n")
    print("| horizon | seed | n origins | persistence | AR | best baseline | ridge | ridge/base | p (paired t) | "
          "p (block boot) | win | AR/persistence p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    report["backtest"] = []
    wins: dict[int, list[bool]] = {}
    for h in cfg.horizons_min:
        for seed in args.seeds:
            t1 = time.time()
            # Inputs screened on this seed's TRAINING rows only (fa.backtest); the window-wide verdicts
            # above are for the table and the live check.
            run, _seed_verdicts = fa.backtest(train_frame, cfg.hygiene, h // g, g, seed)
            if run is None:
                print(f"| {h} min | {seed} | no input passed as a target on this seed's training rows |")
                continue
            base, pick = fa.best_baseline(run)
            cmp_ = fa.compare(run, "ridge_var", base, g, seed)
            ar_vs_p = fa.compare(run, "ar", run.sq["persistence"], g, seed)
            lp = float(np.nanmean(fa.joint_loss(run.sq["persistence"])))
            la = float(np.nanmean(fa.joint_loss(run.sq["ar"])))
            js = fa.joint_surprise(run)
            rest = fa.rest_summary(js.excess, train_frame.index[run.held], seed)
            jsa = fa.joint_surprise(run, calibration=fa.ADAPTIVE)
            rest_a = fa.rest_summary(jsa.excess, train_frame.index[run.held], seed)
            per_t = fa.per_target_table(run, base, pick, g)
            wins.setdefault(h, []).append(cmp_["wins"])
            if cmp_["p_paired_t"] is None or rest.get("n", 0) == 0:
                print(f"| {h} min | {seed} | {cmp_['n']} | too few scored origins (rest n={rest.get('n', 0)}) |")
                report["backtest"].append({"horizon_min": h, "seed": seed, "ridge_vs_best_baseline": cmp_,
                                           "refused_targets": run.refused_targets, "per_target": per_t,
                                           "joint_surprise_heldout": {"n": 0}})
                continue
            print(f"| {h} min | {seed} | {cmp_['n']} | {lp:.4f} | {la:.4f} | {cmp_['loss_reference']:.4f} | "
                  f"{cmp_['loss_candidate']:.4f} | {cmp_['ratio']:.3f} | {cmp_['p_paired_t']:.2g} | "
                  f"{cmp_['p_block_bootstrap']:.2g} | {cmp_['wins']} | {ar_vs_p['p_paired_t']:.2g} |", flush=True)
            report["backtest"].append({"horizon_min": h, "seed": seed, "ridge_vs_best_baseline": cmp_,
                                       "ar_vs_persistence": ar_vs_p, "loss_persistence": lp, "loss_ar": la,
                                       "baseline_pick": pick, "alphas": run.alphas,
                                       "refused_targets": run.refused_targets, "per_target": per_t,
                                       "joint_surprise_heldout": {"n_cal": js.n_cal, "calm_point": js.calm,
                                                                  **_p_shares(js.p), **rest},
                                       "joint_surprise_heldout_adaptive": {"n_cal": jsa.n_cal, "calm_point": jsa.calm,
                                                                           **_p_shares(jsa.p), **rest_a},
                                       "seconds": round(time.time() - t1, 1)})
    # Per-target ratios of the first seed, for the PR table.
    print("\n## Per target (seed %d): ridge / best-baseline loss, Holm p\n" % args.seeds[0])
    print("| target | " + " | ".join(f"{h} min ratio | {h} min p_holm | {h} min base" for h in cfg.horizons_min) + " |")
    print("|---|" + "---|---|---|" * len(cfg.horizons_min))
    by_h = {b["horizon_min"]: {r["target"]: r for r in b["per_target"]} for b in report["backtest"]
            if b["seed"] == args.seeds[0]}
    for t in targets:
        cells = []
        for h in cfg.horizons_min:
            r = by_h.get(h, {}).get(t)
            cells += ([f"{r['ratio']:.3f}", f"{r['p_holm']:.2g}", r["baseline"]] if r else ["-", "-", "-"])
        print(f"| {t} | " + " | ".join(cells) + " |")

    print("\n## Joint surprise on held-out origins (excess = -ln p - calm point; calm = 0)\n")
    print("Calibrated: mean excess CI contains 0; p<=0.05 ~0.05; p<=0.2 ~0.20; hours at rest ~ the null share.\n")
    print("| horizon | seed | calibration | n | n_cal | mean excess | 95% CI | p<=0.05 | p<=0.2 | "
          "hours at rest (calibrated null) | longest hours >0.5 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in report["backtest"]:
        for label, key in (("static", "joint_surprise_heldout"), ("adaptive", "joint_surprise_heldout_adaptive")):
            j = b.get(key) or {}
            if not j.get("n"):
                continue
            print(f"| {b['horizon_min']} min | {b['seed']} | {label} | {j['n']} | {j['n_cal']} | {j['mean_excess']:+.3f} | "
                  f"[{j['mean_ci95'][0]:+.3f}, {j['mean_ci95'][1]:+.3f}] | {j['share_p_le_0_05']:.3f} | "
                  f"{j['share_p_le_0_2']:.3f} | {j['share_hours_at_rest']:.2f} ({j['null_share_hours_at_rest']:.2f}) | "
                  f"{j['longest_hours_above_0_5']} |")

    # 3. Live rest check: train on everything before the last live window, gate, score the window.
    print(f"\n## Live rest check: last {args.live_hours} h, model trained only on data before it\n")
    live_frame = frame.iloc[len(frame) - live_rows:]
    report["live"] = {}
    try:
        bands = {k: verdicts[k].stats["band"] for k in named}
        floors = {k: verdicts[k].stats["min"] for k in named}
        fa.gate_named_inputs(named, live_frame, cfg.hygiene, last_seen, end, bands=bands, floors=floors)
        gate = "pass"
    except fa.HygieneRefused as exc:
        gate = exc.refusals
        print(f"live gate REFUSED: {exc}")
    report["live"]["gate"] = gate
    if gate == "pass":
        print("| horizon | calibration | n | n_cal | mean excess | 95% CI | p<=0.05 | p<=0.2 | "
              "hours at rest (calibrated null) | hourly min..max | longest hours >0.5 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for h in cfg.horizons_min:
            k = h // g
            train_idx = np.arange(0, train_end_i - k)
            held_idx = np.arange(len(frame) - live_rows, len(frame) - k)
            run = fa.run_horizon(frame, targets, inputs, k, g, 0, train_idx=train_idx, held_idx=held_idx)
            for label in (fa.STATIC, fa.ADAPTIVE):
                js = fa.joint_surprise(run, calibration=label)
                rest = fa.rest_summary(js.excess, frame.index[held_idx], 0)
                key = str(h) if label == fa.STATIC else f"{h}_adaptive"
                if rest.get("n", 0) == 0:
                    print(f"| {h} min | {label} | 0 | {js.n_cal} | no scored origins |")
                    report["live"][key] = {"n": 0}
                    continue
                spikes = fa.top_spikes(js, frame.index[held_idx], run.targets)
                shares = _p_shares(js.p)
                report["live"][key] = {"n_cal": js.n_cal, "calm_point": js.calm, **shares, **rest,
                                       "top_spikes": spikes, "mean_shares": fa.mean_shares(js, run.targets)}
                print(f"| {h} min | {label} | {rest['n']} | {js.n_cal} | {rest['mean_excess']:+.3f} | "
                      f"[{rest['mean_ci95'][0]:+.3f}, {rest['mean_ci95'][1]:+.3f}] | {shares['share_p_le_0_05']:.3f} | "
                      f"{shares['share_p_le_0_2']:.3f} | "
                      f"{rest['share_hours_at_rest']:.2f} ({rest['null_share_hours_at_rest']:.2f}) | "
                      f"{rest['hourly_min']:+.2f}..{rest['hourly_max']:+.2f} | "
                      f"{rest['longest_hours_above_0_5']} |")
        for h in [h for h in cfg.horizons_min if report["live"][str(h)].get("n")]:
            print(f"\nTop joint misses, {h} min (readings carrying most of each; a hint, not a cause):")
            for s in report["live"][str(h)]["top_spikes"]:
                carried = ", ".join(f"{c['target']} {c['share']:.0%}" for c in s["carried_by"])
                print(f"- {s['ts']} excess {s['excess']:+.2f}: {carried}")
            shares = report["live"][str(h)]["mean_shares"]
            top = sorted(shares.items(), key=lambda kv: -kv[1])[:5]
            already = sum(shares.get(t, 0.0) for t in ALREADY_SCORED)
            print(f"Average share of the joint score, {h} min: top " +
                  ", ".join(f"{t} {v:.0%}" for t, v in top) +
                  f"; readings node:prediction.body/room already score ({', '.join(ALREADY_SCORED)}): {already:.0%}")

    # 4. Next rung verdict.
    days = common_days(frame, targets)
    ridge_all = all(all(v) for v in wins.values()) and bool(wins)
    neural = "allowed" if ridge_all and days >= NEURAL_MIN_DAYS else "not yet"
    report["ladder"] = {"ridge_beats_baselines_every_seed": {str(h): all(v) for h, v in wins.items()},
                        "common_history_days": days, "neural_min_days": NEURAL_MIN_DAYS, "neural_rung": neural}
    print(f"\n## Ladder\n\nridge beats baselines at every seed: {report['ladder']['ridge_beats_baselines_every_seed']}; "
          f"days with every target present: {days} (need {NEURAL_MIN_DAYS}); small dense network: {neural}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stats = fa.fit_standardizer(train_frame[inputs])
        manifest = fa.build_manifest(cfg, verdicts, stats, (start, end), train_frame.index[-1].to_pydatetime(),
                                     extra={"rung": "ridge_var (eval only, not served)"})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out / 'manifest.json'} and {out / 'report.json'}")
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
