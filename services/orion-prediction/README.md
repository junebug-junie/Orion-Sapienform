# orion-prediction

Forecasts Orion's telemetry, grades every forecast against what really happened, and estimates
cause and effect ("if X happens, how much does the cabinet warm?"). First series: the Athena
cabinet temperature.

Spec: `docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md`.

## What it does, in order

1. **Select** (every 6 h). Backtest every model from 40 past starting points. At each forecast
   distance (5 min, 15 min, 1 h, 6 h, 24 h), pick the model that beats the better of two
   baselines and whose error ranges are calibrated. If none does, serve the baseline there.
2. **Issue** (every completed 5-minute bucket). Fit the chosen models, stitch their forecasts,
   attach error ranges built from their own past errors (split-conformal), and compute the chance
   of crossing each declared threshold (34 °C shed, 32 °C experiment guard). Store it, publish it.
3. **Score**. When the real bucket arrives, write the actual value, the error, and whether it fell
   inside the 80 % and 95 % ranges. A bucket with no readings is scored `missing`, never guessed.
4. **Monitor**. Alerts in `/health` and `/series/{id}/live` (and one log line when they change):
   live coverage out of band, worse than baseline over 24 h, stale input, stale forecast,
   degenerate output (refused, never stored), failed selection.

## Models

| model | role |
|---|---|
| `naive` (same as now), `seasonal_naive` (same time yesterday) | baselines; the better one per horizon is the bar |
| `window_average_1h` | short-horizon smoother |
| `mstl_ets`, `mstl_arima` | MSTL decomposition (cycles up to 1 day) + ETS / ARIMA trend |
| `prophet` | changepoints + every declared seasonality as Fourier terms (incl. weekly, custom) |
| `direct_ar` | one ridge regression per lead time from the temperature's own recent history (`app/direct.py`) |
| `direct_arx` | the same plus the series' covariates (chassis watts, AC watts) as known at the origin -- never a future value |

`direct_ar` vs `direct_arx` is a standing ablation: the scoreboard shows what the covariates add.

A seasonality is used only when training holds 2 full cycles (monthly is refused today). MSTL
skips cycles longer than a day: STL at the weekly period costs ~26 s per fit vs ~2.3 s.

## Cause and effect: three levels, always labelled (`claim_kind`)

| level | endpoint | what it can claim |
|---|---|---|
| `conditional` | `POST /scenario` | "if chassis watts follow this path, temp is forecast to…" -- correlation. Live example: more AC power *goes with* a hotter cabinet, because the AC works harder when it is hot. |
| `observational_effect` | `POST /interventions` then `POST /effects/estimate` | interrupted time series; interval and p-value from placebo runs at earlier times |
| `experimental_effect` | `POST /experiments/design`, log blocks, `POST /experiments/{id}/analyze` | randomised ON/OFF switchback; the only causal answer |

`GET /effects/{action}?series_id=` returns the best available level. `GET /experiments/guard`
says whether it is safe to start an ON block (P(≥ 32 °C within 30 min) < 5 %).

**No dispatcher exists.** Nothing here switches a workload on or off. Whoever performs the action
logs each block (`POST /interventions` with `experiment_id` + `arm`) and re-checks the guard
before each ON block. The 34 °C reflex in hardware-watch stays the hard stop.

## Contracts

- Bus: `orion:prediction:forecast` (`PredictionForecastV1`, `orion/schemas/prediction.py`).
  Consumer: orion-hardware-watch shows the shed probability beside each reflex decision
  (observe-only).
- Tables: `services/orion-sql-db/manual_migration_prediction_v1.sql`.
- Series registry: `config/prediction/series.v1.yaml` (baked into the image).

## Run

```bash
# apply the migration first (additive):
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_prediction_v1.sql
scripts/safe_docker_build.sh orion-prediction up -d --build      # from clean primary main only
curl -fsS http://localhost:8132/health
curl -fsS http://localhost:8132/series/athena.cabinet_temp_c/models
```

## Tests and evals

```bash
cd services/orion-prediction
PYTHONPATH=../..:. python -m pytest tests -q
PYTHONPATH=../..:. python evals/run_causal_calibration_eval.py          # planted effects, many seeds
PYTHONPATH=../..:. python evals/run_cabinet_replay_eval.py --windows 16 # real cabinet history
python evals/export_cabinet_fixture.py --until <iso>                     # refresh the fixture
```
