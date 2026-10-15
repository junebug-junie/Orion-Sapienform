"""Do the causal intervals mean what they say? Many seeds, planted effects, known truth.

- switchback (experimental): 95% interval covers the planted effect in >= 88% of seeds; with no
  effect, p < 0.05 in <= 12% of seeds.
- interrupted time series (observational): placebo 95% interval covers the planted effect in
  >= 80% of seeds (placebo calibration with ~20 placebos is coarse; this guards against it being
  badly wrong, e.g. a dropped sign or a window that includes no action).

    cd services/orion-prediction && PYTHONPATH=../..:. python evals/run_causal_calibration_eval.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app.causal import analyze_switchback, design_switchback, intervention_effect, time_of_day_profile  # noqa: E402
from app.models import build_models, split_seasonalities  # noqa: E402
from tests.helpers import T0, daily_series, spec  # noqa: E402


def switchback_trial(effect: float, seed: int) -> tuple[bool, float]:
    d = design_switchback(T0 + timedelta(days=14), 30, 16, seed=seed)
    s = daily_series(16, amp=1.0, noise=0.1, walk=0.01, seed=seed)
    for b in d["blocks"]:
        if b["arm"] == "on":
            m = (s.index >= pd.Timestamp(b["start"])) & (s.index < pd.Timestamp(b["end"]))
            s[m] += effect * np.minimum(1.0, (np.arange(m.sum()) + 1) / 2)
    prof = time_of_day_profile(s[s.index < pd.Timestamp(d["blocks"][0]["start"])], 5)
    est = analyze_switchback(s, d["blocks"], 10, 5, prof, n_perm=1000, seed=seed)
    return bool(est.ci_lo <= effect <= est.ci_hi), est.p_value


def its_trial(effect: float, seed: int) -> bool | None:
    sp = spec()
    s = daily_series(14, amp=1.0, noise=0.08, walk=0.005, seed=seed)
    start = T0 + timedelta(days=12, hours=int(seed % 20))
    end = start + timedelta(hours=2)
    m = (s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end + timedelta(minutes=30)))
    s[m] += effect
    usable, _ = split_seasonalities(sp, 3 * 288)
    est = intervention_effect(s, build_models(sp, usable)["mstl_ets"], start, end, 5, pre_days=3, lag_min=30,
                              n_placebo=20, seed=seed)
    if est.ci_lo is None:
        return None
    return bool(est.ci_lo <= effect <= est.ci_hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--its-seeds", type=int, default=20)
    args = ap.parse_args()
    failures = []
    cover = [switchback_trial(0.6, s)[0] for s in range(args.seeds)]
    fpr = [switchback_trial(0.0, 1000 + s)[1] < 0.05 for s in range(args.seeds)]
    print(f"switchback: coverage95={np.mean(cover):.2f} (>=0.88)  null FPR={np.mean(fpr):.2f} (<=0.12)")
    if np.mean(cover) < 0.88:
        failures.append("switchback interval under-covers")
    if np.mean(fpr) > 0.12:
        failures.append("switchback null false-positive rate too high")
    its = [its_trial(0.8, s) for s in range(args.its_seeds)]
    known = [x for x in its if x is not None]
    rate = np.mean(known) if known else 0.0
    print(f"ITS: coverage95={rate:.2f} over {len(known)}/{len(its)} trials with an interval (>=0.80)")
    if len(known) < 0.8 * len(its) or rate < 0.80:
        failures.append(f"ITS placebo interval coverage {rate:.2f} on {len(known)} trials")
    print("RESULT:", "PASS" if not failures else "FAIL", *failures, sep="\n  ")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
