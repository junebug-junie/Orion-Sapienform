"""House-meter heat balance: does the central AC switch on above ~75 F outside? (read-only, live DB)

    python services/orion-prediction/evals/run_house_heat_balance_eval.py [--threshold 75] [--lag-h 3]

The Rocky Mountain Power meter (energy_usage_interval, hourly, whole kWh, ~3 days behind) minus the
known loads -- the cabinet machines (orion_biometrics_cluster per node) and the portable AC
(home_cooling_sample) -- leaves household + central AC. Regressed on outdoor F above the threshold,
`lag-h` hours earlier (prediction_weather_hourly), with HAC standard errors.

Not a CI gate: it needs the live database and changes with the season. It is the check behind the
`athena.central_ac_proxy` covariate (config/prediction/series.v1.yaml); a coefficient that collapses to
~0 in summer data would retire it. 2026-10-20 result: +0.33 kWh/h per F above 75 F (p ~1e-18).
"""
from __future__ import annotations

import argparse
import io
import subprocess

import numpy as np
import pandas as pd

SQL = """COPY (
WITH h AS (SELECT interval_start AS t, energy_kwh AS house FROM energy_usage_interval WHERE source = 'rockymountain_power'),
n AS (SELECT date_trunc('hour', observed_at) AS t,
             avg((measurements_by_node->'athena'->>'chassis_watts')::float) AS athena,
             avg((measurements_by_node->'circe'->>'chassis_watts')::float) AS circe,
             coalesce(avg((measurements_by_node->'hecate'->>'chassis_watts')::float), 0) AS hecate
      FROM orion_biometrics_cluster GROUP BY 1),
a AS (SELECT date_trunc('hour', ts) AS t, avg(cooling_watts) AS ac FROM home_cooling_sample WHERE role = 'cabinet_cooling' GROUP BY 1),
w AS (SELECT hour_ts + interval '{lag} hours' AS t, temp_f FROM prediction_weather_hourly WHERE NOT is_forecast)
SELECT h.t, h.house, n.athena, n.circe, n.hecate, a.ac, w.temp_f FROM h JOIN n USING (t) JOIN a USING (t) JOIN w USING (t)
ORDER BY 1) TO STDOUT WITH CSV HEADER"""


def main() -> int:
    import statsmodels.api as sm

    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=75.0)
    ap.add_argument("--lag-h", type=int, default=3)
    ap.add_argument("--container", default="orion-athena-sql-db")
    args = ap.parse_args()
    out = subprocess.run(["docker", "exec", "-i", args.container, "psql", "-U", "postgres", "-d", "conjourney",
                          "-v", "ON_ERROR_STOP=1", "-c", SQL.format(lag=args.lag_h)], capture_output=True, text=True,
                         check=True).stdout
    df = pd.read_csv(io.StringIO(out)).dropna()
    if df.empty:
        print("no overlapping hours (is prediction_weather_hourly backfilled?)")
        return 1
    df["rest"] = df.house - (df.athena + df.circe + df.hecate + df.ac) / 1000
    df["above"] = (df.temp_f - args.threshold).clip(lower=0)
    print(f"hours={len(df)} {df.t.iloc[0]} -> {df.t.iloc[-1]}; mean house {df.house.mean():.2f} = known "
          f"{(df.house - df.rest).mean():.2f} + household/central {df.rest.mean():.2f} kWh/h")
    print(df.groupby(pd.cut(df.temp_f, [0, 50, 60, 65, 70, 75, 80, 85, 90, 120]), observed=True).rest.agg(["mean", "count"]).round(2).to_string())
    fit = sm.OLS(df.rest, sm.add_constant(df[["above"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print(f"\nhousehold+central kWh/h per F above {args.threshold:g} ({args.lag_h} h earlier): "
          f"{fit.params.above:.3f} (se {fit.bse.above:.3f}, p={fit.pvalues.above:.2g}); hours above: {(df.above > 0).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
