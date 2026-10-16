"""Backtest the full model zoo on REAL cabinet history and check the served blend is honest.

Fixture: evals/fixtures/cabinet_5min.csv.gz (export_cabinet_fixture.py; 5-min means from production,
2026-09-26 AC install -> 2026-10-15). Same code production runs: app.backtest + app.models, the repo
registry entry for athena.cabinet_temp_c, no database.

Checks (exit 1 on failure):
- monthly seasonality is refused on this history; daily is used.
- every model produces residuals on >= 80% of windows (a model that silently fails is caught).
- the served champion blend never loses to the better baseline at any horizon (rel_mae <= 1).
- the blend's leave-one-out 80% coverage sits in [0.65, 0.95] at every horizon.
- at least one horizon is served by a non-baseline model, or the report says the baseline is served.
Also reports (not gated): a conditional scenario for +200 W chassis power and the AC covariate.

    cd services/orion-prediction && PYTHONPATH=../..:. python evals/run_cabinet_replay_eval.py   # 40 windows, as production
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
sys.path[:0] = [str(SERVICE), str(REPO)]

from app.backtest import horizon_metrics, run_backtest, select_champion, step_owner, stitch  # noqa: E402
from app.causal import CausalError, scenario  # noqa: E402
from app.frame import to_grain, training_slice  # noqa: E402
from app.models import BASELINES, build_models, split_seasonalities  # noqa: E402
from app.series import load_registry  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cabinet_5min.csv.gz"


def load(col: str, spec, fill: bool = True) -> pd.Series:
    with gzip.open(FIXTURE, "rt") as fh:
        df = pd.read_csv(fh)
    ts = pd.to_datetime(df["bucket_start"], utc=True)
    return to_grain(zip(ts, df[col].astype(float)), spec.grain_min, spec.agg, spec.max_fill_min if fill else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--step-min", type=int, default=180)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    reg = load_registry(REPO / "config" / "prediction" / "series.v1.yaml")
    spec = reg.get("athena.cabinet_temp_c")
    y = load("cabinet_temp_c", spec)
    actuals = load("cabinet_temp_c", spec, fill=False).reindex(y.index)
    cov_cols = {"athena.chassis_watts": "chassis_watts", "athena.cabinet_cooling_watts": "cooling_watts"}
    covariates = pd.DataFrame({cid: load(cov_cols[cid], reg.get(cid), fill=False).reindex(y.index) for cid in spec.covariates},
                              index=y.index)
    train = training_slice(y, spec)
    usable, refused = split_seasonalities(spec, len(train))
    zoo = build_models(spec, usable)
    step = args.step_min // spec.grain_min
    results, timing = {}, {}
    for name in zoo:
        t0 = time.monotonic()
        results[name] = run_backtest(y, spec, name, lambda u, n=name: build_models(spec, u)[n], args.windows, step,
                                     actuals=actuals, covariates=covariates)
        timing[name] = time.monotonic() - t0
    sel = select_champion(results, spec)
    h = spec.max_horizon_steps
    res = {n: r.residuals for n, r in results.items()}
    blend = stitch(res, step_owner(sel.assignment, spec.grain_min, h))
    blend_m = {str(hm): horizon_metrics(blend, spec.steps(hm) - 1) for hm in spec.horizons_min}

    failures = []
    if "monthly" not in refused or "daily" not in {s.name for s in usable}:
        failures.append(f"seasonality split wrong: usable={[s.name for s in usable]} refused={refused}")
    for name, r in results.items():
        if r.n_windows < 0.8 * len(r.origins):
            failures.append(f"{name} produced residuals on only {r.n_windows}/{len(r.origins)} windows: {r.errors[:2]}")
    for hm in spec.horizons_min:
        key = str(hm)
        ref = min(sel.metrics[b][key]["mae"] for b in BASELINES)
        bm = blend_m[key]
        if bm["mae"] is None or bm["mae"] > ref + 1e-9:
            failures.append(f"{hm}min: blend MAE {bm['mae']} worse than best baseline {ref}")
        if bm["coverage80"] is None or not (0.65 <= bm["coverage80"] <= 0.95):
            failures.append(f"{hm}min: blend LOO coverage80 {bm['coverage80']}")

    print(f"history buckets={len(y)} train={len(train)} usable={[s.name for s in usable]} refused={refused}")
    print(f"windows={args.windows} step={args.step_min}min\n")
    hdr = "model".ljust(20) + "".join(f"{hm:>16}min" for hm in spec.horizons_min) + "   seconds"
    print(hdr + "\n" + "-" * len(hdr))
    for name in results:
        cells = []
        for hm in spec.horizons_min:
            m = sel.metrics[name][str(hm)]
            mark = "*" if sel.assignment[hm] == name else " "
            cells.append(f"{m['mae'] or float('nan'):6.3f}/{m['rel_mae'] or float('nan'):4.2f}{mark}".rjust(19))
        print(name.ljust(20) + "".join(cells) + f"{timing[name]:10.1f}")
    print("\ncells: MAE C / MAE relative to the better baseline; * = serves that horizon")
    print("blend:", {hm: f"MAE {blend_m[str(hm)]['mae']:.3f} cov80 {blend_m[str(hm)]['coverage80']:.2f}"
                     for hm in spec.horizons_min})
    print("assignment:", sel.assignment, "served_baseline:", sel.served_baseline)

    scen = {}
    for cov_id in spec.covariates:
        cov_spec = reg.get(cov_id)
        col = {"athena.chassis_watts": "chassis_watts", "athena.cabinet_cooling_watts": "cooling_watts"}[cov_id]
        x = load(col, cov_spec).reindex(y.index)
        # End where the covariate's own history ends (the fixture's AC tail can be empty).
        last = x.last_valid_index()
        stop = y.index.get_loc(last) + 1 if last is not None else len(y)
        tail = slice(stop - 7 * 288, stop)
        try:
            hold = float(x.iloc[tail].dropna().iloc[-12:].mean())
            r = scenario(y.iloc[tail], x.iloc[tail], cov_id, [hold + 200.0] * 24, spec.grain_min)
            scen[cov_id] = {"long_run_C_per_100": 100 * r.long_run_effect_per_unit, "delta_2h_C": r.delta[-1]}
        except CausalError as exc:
            scen[cov_id] = {"error": str(exc)}
    print("\nconditional scenarios (+200 units for 2h; correlation, not cause):")
    for k, v in scen.items():
        print(f"  {k}: {json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})}")

    if args.json:
        args.json.write_text(json.dumps({"metrics": sel.metrics, "assignment": sel.assignment, "blend": blend_m,
                                         "timing": timing, "scenarios": scen, "failures": failures}, default=str))
    print("\nRESULT:", "PASS" if not failures else "FAIL")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
