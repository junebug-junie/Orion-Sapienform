"""Family C end to end on planted truth: switchback -> effect table -> efference copy -> miss score.

Each seed: a reading with a daily cycle and AR(1) noise; a 12-block randomised switchback plants an
effect; the table is built from it (app/family_c.rows_from_experiment); then 40 later acts are
forecast by a persistence-style forecaster whose 9 quantiles (as Family A stores them) are the
empirical 15-min changes of the pre-experiment history, and each act's 15-min reading is scored against the efference copy.

Checks (nominal values in brackets):
- table 95 % interval covers the planted effect in >= 88 % of seeds [95 %];
- table right: efference mean excess within 0.20 of 0 [0], share p <= 0.05 in 0.02..0.10 [0.05],
  and the gate prefers the efference copy over the forecast alone (gap > 0, prob_better >= 0.9);
- null actuator (no planted effect): gap within 2 SE of 0 (knowing "no effect" costs nothing);
- wrong-sign table: mean excess > 0.5 and gap < 0 (a broken self-model is visible as surprise).

    cd services/orion-prediction && PYTHONPATH=../..:. python evals/run_family_c_synthetic_eval.py

Exit 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app import family_c as fc  # noqa: E402
from app.causal import design_switchback  # noqa: E402

GRAIN = 5
H = 15
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    daily = np.sin(2 * np.pi * (idx * GRAIN % 1440) / 1440)
    ar = np.zeros(n)
    e = rng.normal(0, 0.06, n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    return 26 + daily + ar


def trial(effect: float, seed: int, table_sign: float = 1.0, n_acts: int = 40) -> tuple[bool, list[dict]]:
    exp_start = T0 + timedelta(days=fc.PROFILE_DAYS)
    design = design_switchback(exp_start, 20, 12, seed=seed)
    exp = {"experiment_id": f"s{seed}", "action": "synthetic.act", "burn_in_min": 10, "design": design}
    # history + experiment + 2 days of later acts
    n = int((fc.PROFILE_DAYS + 3) * 1440 / GRAIN)
    idx = pd.date_range(T0, periods=n, freq=f"{GRAIN}min", tz="UTC")
    y = _series(n, seed)
    for b in design["blocks"]:
        if b["arm"] == "on":
            m = (idx >= pd.Timestamp(b["start"]) + pd.Timedelta(minutes=10)) & (idx < pd.Timestamp(b["end"]))
            y[m] += effect
    frame = pd.DataFrame({"r": y}, index=idx)
    last = max(pd.Timestamp(b["end"]) for b in design["blocks"])
    (row,) = fc.rows_from_experiment(exp, design["blocks"], frame[frame.index < last], {"r": "C"}, GRAIN, n_perm=500)
    covered = row.ci_lo <= effect <= row.ci_hi
    table = [replace(row, estimate=table_sign * row.estimate)]
    # Forecaster: h-step change of the same series on the history, as an empirical 80 % range.
    hist = pd.Series(y, index=idx)[idx < pd.Timestamp(exp_start)]
    k = H // GRAIN
    d = (hist.shift(-k) - hist).dropna().to_numpy()
    levels = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
    dq = np.quantile(d, levels)
    rng = np.random.default_rng(seed + 7)
    scores = []
    after = np.where(idx >= last + pd.Timedelta(hours=1))[0]
    for i, o in enumerate(rng.choice(after[:-k - 1], n_acts, replace=False)):
        origin = idx[o]
        act_start = origin
        target_ts = origin + pd.Timedelta(minutes=H)
        actual = y[o + k] + effect   # the act moves this reading by the true effect
        f = {"target": "r", "origin_ts": origin, "target_ts": target_ts, "point": float(y[o]),
             "quantiles": {f"{lv:g}": float(y[o] + v) for lv, v in zip(levels, dq)}}
        eff = fc.efference_copy(f"{seed}-{i}", "synthetic.act", act_start.to_pydatetime(), f, table)
        scores.append(fc.score(eff, actual))
    return covered, scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args(argv)
    ok = True

    cov, right = 0, []
    for s in range(args.seeds):
        c, sc = trial(0.4, s)
        cov += c
        right += sc
    rate = cov / args.seeds
    g = fc.gate(right)
    print(f"table right  : CI coverage {rate:.2f}  excess {g['mean_excess']:+.3f}  p<=.05 {g['share_p_le_0_05']:.3f}  "
          f"gap {g['gap']:+.3f} se {g['se']:.3f} prob_better {g['prob_better']:.3f}  acts {g['n_acts']}")
    ok &= rate >= 0.88 and abs(g["mean_excess"]) <= 0.20 and 0.02 <= g["share_p_le_0_05"] <= 0.10
    ok &= g["gap"] > 0 and g["prob_better"] >= 0.9

    null = [x for s in range(args.seeds) for x in trial(0.0, 1000 + s)[1]]
    gn = fc.gate(null)
    print(f"null act     : excess {gn['mean_excess']:+.3f}  gap {gn['gap']:+.3f} se {gn['se']:.3f} "
          f"prob_better {gn['prob_better']:.3f}")
    ok &= abs(gn["gap"]) <= 2 * gn["se"] + 0.02

    wrong = [x for s in range(args.seeds) for x in trial(0.4, 2000 + s, table_sign=-1.0)[1]]
    gw = fc.gate(wrong)
    print(f"wrong sign   : excess {gw['mean_excess']:+.3f}  gap {gw['gap']:+.3f} se {gw['se']:.3f} "
          f"prob_better {gw['prob_better']:.3f}")
    ok &= gw["mean_excess"] > 0.5 and gw["gap"] < 0

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
