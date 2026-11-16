"""Family C on live data, read-only. Nothing is written anywhere.

1. Effect table: rebuilt from every finished (``analyzed``) randomised experiment in
   prediction_experiment, over every raw reading Family A reads (app/family_c.EffectTable with the
   Family A loader). Printed, sorted by |estimate| / SE.
2. Rest check of the efference miss score: with no act (effect 0, SE 0) the efference copy is the
   Family A reading forecast itself, so the score's calm point is Family A's per-reading calibration
   read through ``family_c.pit`` (its 9 stored quantiles, Gaussian tails). The Gaussian read-off of
   the 80 % range alone is printed beside it for comparison. Over the last --hours of scored champion rows:
   mean excess (2-hour block bootstrap 95 % interval), share p <= 0.05 (0.05 when calibrated), and
   the share of hours at rest (hourly mean excess <= 0).
3. Planted-act replay on real forecasts: at --acts random scored rows per horizon, the real reading
   is moved by 2 forecast sigmas (a planted act); scored against a table that knows the effect vs the
   forecast alone. Shows the scorer separates "I did X and it did what I expected" from "surprise"
   on real forecast ranges, not only synthetic ones.

    cd services/orion-prediction
    POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_c_live_eval.py --hours 48

Exit 0 always (a report); the verdict lines say what moved.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3])]

from app import family_a as fa  # noqa: E402
from app import family_c as fc  # noqa: E402

CONFIG = HERE.parents[3] / "config" / "prediction" / "family_a.v1.yaml"


def _rows(conninfo: str, sql: str, args: tuple) -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(conninfo, autocommit=False, row_factory=dict_row) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        return conn.execute(sql, args).fetchall()


def effect_table(conninfo: str, cfg) -> dict:
    from app.family_a_load import load_frame
    from app.store import PostgresStore

    table = fc.EffectTable(PostgresStore(conninfo),
                           lambda s, e: load_frame(conninfo, cfg.inputs, cfg.grain_min, s, e),
                           units={i.id: i.unit for i in cfg.inputs}, grain_min=cfg.grain_min)
    return table.build()


def _fc(r: dict) -> dict:
    return {"target": r["series_id"], "origin_ts": r["origin_ts"], "target_ts": r["target_ts"],
            "point": r["point"], "quantiles": r["quantiles"]}


def _null_eff(r: dict) -> fc.Efference:
    return fc.null_efference(str(r["forecast_id"]), _fc(r))


def _block_ci(df: pd.DataFrame, seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    blocks = [g["excess"].to_numpy() for _, g in df.groupby(df["target_ts"].dt.floor("2h"))]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        means.append(float(np.concatenate([blocks[i] for i in pick]).mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48)
    ap.add_argument("--acts", type=int, default=300)
    ap.add_argument("--skip-table", action="store_true")
    args = ap.parse_args(argv)
    conninfo = os.environ["POSTGRES_URI"]
    cfg = fa.parse_config(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))

    if not args.skip_table:
        t = effect_table(conninfo, cfg)
        print(f"== effect table: {len(t['experiments'])} analyzed experiment(s), {t['n_rows']} rows, "
              f"{t['n_with_estimate']} with an estimate; errors {t['errors'] or 'none'}")
        rows = sorted((r for r in t["rows"] if r["estimate"] is not None),
                      key=lambda r: -abs(r["estimate"]) / max(r["se"] or 1e-12, 1e-12))
        for r in rows:
            print(f"  {r['actuator']:<22} {r['target']:<32} +{r['window_start_min']}..{r['window_end_min']} min  "
                  f"{r['estimate']:+9.3f} {r['unit'] or '':<4} 95% {r['ci_lo']:+9.3f}..{r['ci_hi']:+9.3f}  "
                  f"se {r['se']:.3f}  p {r['p_value']:.3f}  on/off {r['n_on']}/{r['n_off']}"
                  f"{'' if r['profile_adjusted'] else '  (no profile)'}")
        for r in t["rows"]:
            if r["estimate"] is None:
                print(f"  {r['actuator']:<22} {r['target']:<32} no estimate: {r['reason']}")

    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    data = _rows(conninfo,
                 "SELECT forecast_id, series_id, horizon_min, origin_ts, target_ts, point, quantiles, actual, in_80 "
                 "FROM prediction_forecast WHERE starts_with(series_id, 'family_a:') AND role = 'champion' "
                 "AND score_status = 'scored' AND target_ts >= %s", (since,))
    print(f"\n== rest check (no act): {len(data)} scored Family A reading forecasts since {since:%Y-%m-%d %H:%M}Z")
    scored = []
    for r in data:
        try:
            s = fc.score(_null_eff(r), r["actual"])
            z = (float(r["actual"]) - float(r["point"])) / fc.sigma_from_quantiles(r["quantiles"])
        except fc.EfferenceRefused:
            continue
        gauss = -math.log(max(math.erfc(abs(z) / math.sqrt(2.0)), fc.P_FLOOR)) - fc.CALM_SURPRISAL
        scored.append({**s, "horizon": r["horizon_min"], "series_id": r["series_id"], "in_80": r["in_80"],
                       "target_ts": pd.Timestamp(r["target_ts"]), "gauss_excess": gauss})
    df = pd.DataFrame(scored)
    if df.empty:
        print("  no rows")
        return 0
    df = df[df["status"] == "scored"]
    for h, g in df.groupby("horizon"):
        lo, hi = _block_ci(g)
        hourly = g.groupby(g["target_ts"].dt.floor("1h"))["excess"].mean()
        print(f"  h={h:>2}min n={len(g):>6}  mean excess {g['excess'].mean():+.3f} (95% {lo:+.3f}..{hi:+.3f})  "
              f"p<=.05 {(g['p'] <= 0.05).mean():.3f}  in_80 {g['in_80'].mean():.3f}  "
              f"hours at rest {(hourly <= 0).mean():.2f}  median excess {g['excess'].median():+.3f}  "
              f"[Gaussian-80% read-off: mean excess {g['gauss_excess'].mean():+.3f}]")
    worst = (df.groupby("series_id")["excess"].mean().sort_values(ascending=False))
    print("  highest mean excess by reading: " + ", ".join(f"{k.split(':', 1)[1]} {v:+.2f}" for k, v in worst.head(5).items()))
    print("  lowest:                         " + ", ".join(f"{k.split(':', 1)[1]} {v:+.2f}" for k, v in worst.tail(3).items()))

    print(f"\n== planted-act replay on real forecasts ({args.acts} acts per horizon, act = +2 forecast sigma)")
    rng = np.random.default_rng(0)
    by_id = {str(r["forecast_id"]): r for r in data}
    for h in sorted(df["horizon"].unique()):
        ids = df[df["horizon"] == h]["act_id"].to_numpy()
        pick = rng.choice(ids, min(args.acts, len(ids)), replace=False)
        knows, blind = [], []
        for fid in pick:
            r = by_id[str(fid)]
            base = _null_eff(r)
            sigma = fc.sigma_from_quantiles(base.baseline_quantiles)
            delta = 2.0 * sigma
            row = fc.EffectRow("planted", r["series_id"], None, 0, int(h) + 1, fc.METHOD_RANDOMISED, delta,
                               None, None, 0.25 * sigma, 6, 6, None, ("planted",))
            act_start = r["origin_ts"] + timedelta(minutes=cfg.grain_min)   # origin bucket complete first
            eff = fc.efference_copy(str(fid), "planted", act_start, r["target_ts"], _fc(r), [row], cfg.grain_min)
            actual = float(r["actual"]) + delta
            knows.append(fc.score(eff, actual))
            blind.append(fc.score(_null_eff(r), actual))
        gk = fc.gate(knows)
        print(f"  h={h:>2}min  table knows the act: excess {gk['mean_excess']:+.3f}  p<=.05 {gk['share_p_le_0_05']:.3f}  "
              f"gap vs forecast alone {gk['gap']:+.3f} (se {gk['se']:.3f}, prob_better {gk['prob_better']:.3f})")
        gb = fc.gate(blind)
        print(f"  h={h:>2}min  no table (blind):    excess {gb['mean_excess']:+.3f}  p<=.05 {gb['share_p_le_0_05']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
