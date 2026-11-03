"""Heat per watt (app/heat_balance.py): planted slope, null, and an unseen machine, on simulated hourly
data with autocorrelated noise; optional live read-only fit.

    cd services/orion-prediction && PYTHONPATH=../..:. python evals/run_heat_per_watt_eval.py [--draws 200]
    ... --live [--experiment-id <uuid>]   # read-only fit on the real DB (POSTGRES_URI), nothing stored

Checks (exit 1 on failure), each at 120 h (5 days, the window the live fit uses) with AR(1) noise:
- planted athena slope 0.43 C/100 W: the 95% interval covers it in >= 85% of draws (measured ~89%:
  feasible GLS + HAC is still under nominal on short, persistent series -- plain OLS+HAC was ~82%), and
  the mean estimate is within 10% of the truth.
- planted null (no slope): a nonzero claim (fit not refused AND interval excludes 0) in <= 12% of draws.
- circe (watts the sensor does not see, slope 0) next to a real athena slope: interval covers 0 in >= 85%.
- hecate (on for 2 of 120 hours) is refused as flat in every draw, never fitted.
- total fit (cabinet with the room free; the room also takes 0.9 / 0.3 C per 100 W of athena / circe and
  follows outdoor air): its intervals cover local + room slopes in >= 85% of draws for both machines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
sys.path[:0] = [str(SERVICE), str(REPO)]

from app.heat_balance import HEAT_BALANCE, HeatBalanceRefused, fit_from_frame  # noqa: E402

CFG = HEAT_BALANCE["athena.cabinet_temp_c"]
T = datetime(2026, 10, 29, tzinfo=timezone.utc)
SLOPE = 0.0043
ROOM_A, ROOM_C = 0.009, 0.003   # machine heat that mixes into the room (C per W): the total fit's extra


def frame(seed: int, slope_a: float, n: int = 120, phi: float = 0.6, noise: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = 450 + 150 * (np.sin(np.arange(n) / 5.0 + rng.uniform(0, 6)) > 0.3) + rng.normal(0, 30, n)
    c = 1000 + rng.normal(0, 200, n)
    h = np.zeros(n)
    h[10:12] = 500.0
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, noise)
    gap = -1.3 + slope_a * a + e
    outdoor = 50 + 6 * np.sin(2 * np.pi * np.arange(n) / 24 + rng.uniform(0, 6)) + rng.normal(0, 0.5, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, noise)
    room = 10 + 0.2 * outdoor + ROOM_A * a + ROOM_C * c + r
    return pd.DataFrame({"gap": gap, "cabinet": room + gap, "outdoor.temp_f": outdoor, "athena.chassis_watts": a,
                         "circe.chassis_watts": c, "hecate.chassis_watts": h,
                         "athena.cabinet_cooling_watts": 118.0 + rng.normal(0, 0.3, n)},
                        index=pd.date_range(T, periods=n, freq="60min"))


def simulate(draws: int) -> tuple[dict, list[str]]:
    cover_a = cover_c = claims_null = refused_null = hecate_fitted = cover_ta = cover_tc = 0
    est = []
    for s in range(draws):
        fit = fit_from_frame(frame(s, SLOPE), CFG, "athena.cabinet_temp_c", T, T)
        a, c = fit.slopes["athena.chassis_watts"], fit.slopes["circe.chassis_watts"]
        cover_a += a["ci_lo"] <= SLOPE <= a["ci_hi"]
        cover_c += c["ci_lo"] <= 0 <= c["ci_hi"]
        hecate_fitted += "hecate.chassis_watts" in fit.slopes
        ta, tc = fit.total["slopes"]["athena.chassis_watts"], fit.total["slopes"]["circe.chassis_watts"]
        cover_ta += ta["ci_lo"] <= SLOPE + ROOM_A <= ta["ci_hi"]
        cover_tc += tc["ci_lo"] <= ROOM_C <= tc["ci_hi"]
        est.append(a["per_w"])
        try:
            nf = fit_from_frame(frame(10_000 + s, 0.0, noise=0.3), CFG, "athena.cabinet_temp_c", T, T)
            na = nf.slopes["athena.chassis_watts"]
            claims_null += not (na["ci_lo"] <= 0 <= na["ci_hi"])
        except HeatBalanceRefused:
            refused_null += 1
    out = {"draws": draws, "planted_coverage": cover_a / draws, "planted_mean_per_100w": 100 * float(np.mean(est)),
           "circe_covers_zero": cover_c / draws, "null_claim_rate": claims_null / draws,
           "null_refused_rate": refused_null / draws, "hecate_fitted": hecate_fitted,
           "total_athena_coverage": cover_ta / draws, "total_circe_coverage": cover_tc / draws}
    failures = []
    if out["planted_coverage"] < 0.85:
        failures.append(f"planted coverage {out['planted_coverage']:.2f} < 0.85")
    if abs(np.mean(est) - SLOPE) > 0.1 * SLOPE:
        failures.append(f"planted mean {100 * np.mean(est):.3f} C/100 W is > 10% off {100 * SLOPE:.2f}")
    if out["null_claim_rate"] > 0.12:
        failures.append(f"null claim rate {out['null_claim_rate']:.2f} > 0.12")
    if out["circe_covers_zero"] < 0.85:
        failures.append(f"circe interval covers 0 in only {out['circe_covers_zero']:.2f}")
    for k in ("total_athena_coverage", "total_circe_coverage"):
        if out[k] < 0.85:
            failures.append(f"{k} {out[k]:.2f} < 0.85")
    if hecate_fitted:
        failures.append(f"hecate fitted in {hecate_fitted} draws (should be refused as flat)")
    return out, failures


def live(experiment_id: str | None, days: float) -> dict:
    """Read-only: every store write raises."""
    from app.effects import Effects
    from app.engine import Engine
    from app.series import load_registry
    from app.store import PostgresStore

    class ReadOnly:
        def __init__(self, st):
            self.st = st

        def __getattr__(self, k):
            if k.startswith(("insert", "write", "set_", "upsert", "prune", "update", "delete")):
                raise RuntimeError(f"read-only eval: {k} blocked")
            return getattr(self.st, k)

    reg = load_registry(REPO / "config" / "prediction" / "series.v1.yaml")
    eng = Engine(reg, ReadOnly(PostgresStore(os.environ["POSTGRES_URI"])), backtest_windows=40, backtest_step_min=180,
                 select_interval_sec=21600, settle_sec=60, publish_enabled=False)
    return Effects(eng).heat_per_watt("athena.cabinet_temp_c", days, experiment_id, False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--experiment-id")
    ap.add_argument("--days", type=float, default=5.0)
    args = ap.parse_args()
    out, failures = simulate(args.draws)
    print(json.dumps(out, indent=1))
    if args.live:
        print(json.dumps(live(args.experiment_id, args.days), indent=1, default=str))
    print("\nRESULT:", "PASS" if not failures else "FAIL")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
