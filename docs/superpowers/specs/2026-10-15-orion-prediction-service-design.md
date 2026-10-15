# orion-prediction: forecasting, evaluation, and "what if" service

Status: DESIGN (2026-10-15). No code yet. Phases 1–2 are ordinary service work.
Phase 3 touches autonomy (Orion runs experiments on its own hardware) and needs
proposal-mode sign-off before implementation.

## Arsonist summary

Do not build "a Prophet service". Build a **forecast contract plus a scoring
loop**. Models are cheap and replaceable; what has lasting value is:

1. one tidy input shape (`series_id, ts, y` + covariates),
2. one forecast output shape (point + quantiles per horizon step),
3. a backtest gate that a model must pass to be used, and
4. a live reducer that scores every stored forecast when the real value arrives.

Without (3) and (4), a forecast is just a confident-looking line. With them,
Orion gets a real, honest prediction-error signal for the first time — a value
it predicted, a value that happened, and the gap between them.

The "if Juniper focuses on X, the room gets Y warmer" goal is **not a forecast**.
A forecast says what will happen if things continue as they usually do. A "what
if" asks what changes when someone acts. Historical correlation answers this
badly here (see data facts: hourly chassis watts vs cabinet temp correlates at
only 0.15, because the AC and the time of day confound it). The honest route is
an action log plus small, safe experiments Orion runs itself. That is phase 3.

## Data facts (pulled live 2026-10-15, not assumed)

Source: `public.orion_biometrics_summary`, `node='athena'`, JSONB `measurements`.

| fact | value | design consequence |
|---|---|---|
| cadence | median 31 s | resample to 1 min / 5 min / 1 h grains; never fit raw |
| history | 2026-08-29 → now, **48 days**, 124k rows | **monthly seasonality cannot be estimated** (needs ≥2–3 cycles). Weekly has ~7 cycles: usable but weak |
| gaps | 3 bad days (09-01: 709 rows, 09-12: 812, 09-16: 360); max gap 24 h | explicit gap policy; gaps are masked, never interpolated across > N min |
| daily rhythm | cabinet temp mean 29.9 °C at 10:00 MDT, 27.7 °C at 19:00 MDT | real diurnal signal, ~2 °C swing → daily seasonality is earned |
| weekly | one day (ISO Thursday) holds all 271 readings ≥ 34 °C | single incident, not a weekly pattern; would fool a weekly term → needs an event/outlier flag |
| `timestamp` column | `varchar`, not `timestamptz` | adapter must cast; index is `(node, timestamp)` on the string |
| PM sensor keys | absent from the last 24 h | series registry must tolerate a sensor dropping out; the scorer must not read "no data" as "calm" |
| AC plug | `home_cooling_sample`, 336k rows since **2026-09-26** only | AC data is a covariate only for the last 3 weeks; also likely a **structural break** on 09-26 |
| control loop | hardware-watch sheds GPU load at ≥ 34 °C | the series is **controlled**: high temperatures are clipped by Orion's own reflex. A forecaster that ignores this will under-predict what would happen without the reflex |

## Current architecture

- No forecasting code exists. `statsmodels`/`prophet` are not imported anywhere.
  `orion/energy/*` uses the word "forecast" for bill projection only.
- Cabinet path: Nano ESP32 → `scripts/orion_cabinet_sensor_reader.py` →
  `/run/orion-sensors/latest.json` → orion-biometrics → `orion_biometrics_summary`
  (raw `measurements` + baseline-relative `pressures`).
- `orion/substrate/prediction_error_magnitude.py` turns a node's reading history
  into a percentile against its own past. It is *surprise against history*, not
  against a prediction. A real forecast residual is a stronger signal for the
  same consumer.
- `orion-hardware-watch` reacts to cabinet heat **after** it crosses 34 °C.
- `orion-self-experiments` validates experiments but has never dispatched one
  (see memory: "self-experiments are theater"). Phase 3 must not repeat that.
- `orion-analytics` (dbt + Lightdash) reads Postgres for dashboards. It is a
  reporting surface, not a model host.

## Missing questions (for Juniper)

1. **"Room" vs "cabinet".** The only temperature sensor is inside the cabinet.
   Do you mean the room? If yes, a room sensor is a hardware prerequisite for
   the phase 3 goal.
2. **What is "focus on X" physically?** A GPU workload (which lane), a chat
   session, you being in the room? The experiment needs a named, loggable action.
3. **First consumer.** Recommended: hardware-watch shedding *before* 34 °C based
   on `P(temp ≥ 34 within 30 min)`. Confirm, or name another.
4. **Horizons.** Proposed: nowcast (0), 15 min, 1 h, 6 h, 24 h. Anything longer
   is not supported by 48 days of data.

## Model suite

One interface for every model: `fit(frame, spec) -> FittedModel`,
`predict(h, future_covariates, quantiles) -> ForecastFrame`. Library choice:
the Nixtla stack (`statsforecast`, `mlforecast`, `utilsforecast`) plus
`prophet` and `statsmodels`. Nixtla gives one shared dataframe shape
(`unique_id, ds, y`), fast AutoARIMA/ETS/MSTL, and a cross-validation routine
that already does rolling-origin backtests. `pmdarima` is skipped (slow,
weakly maintained; `statsforecast.AutoARIMA` replaces it).

| tier | model | why it is here | phase |
|---|---|---|---|
| baseline (mandatory) | naive, seasonal naive (24 h), window mean | every other model must beat these, or it is not used | 1 |
| statistical | AutoETS, AutoARIMA, **MSTL** (multiple seasonalities: daily + weekly + custom) | strong, fast, honest intervals | 1 |
| statistical | Prophet | changepoints, holidays/custom events, custom seasonalities with Fourier order, additive regressors | 1 |
| covariates | SARIMAX / ARIMAX (statsmodels) | univariate target + exogenous drivers (chassis watts, AC watts, GPU load) | 2 |
| multivariate | VAR / VARMAX | joint forecasts where variables drive each other (temp ↔ humidity ↔ fan) | 2 |
| machine learning | LightGBM via `mlforecast` (lags, rolling stats, calendar, covariates; direct multi-horizon) | nonlinear covariate effects; one global model across many series | 2 |
| state space | unobserved-components / Kalman (statsmodels) | **nowcast** the current value when the sensor is stale or noisy | 2 |
| foundation | Chronos-Bolt or TimesFM, zero-shot, GPU | challenger only; must earn its slot through the same backtest gate | 4 |

### Bells and whistles, as concrete features

- **Seasonality**: hourly (sub-daily, for 1-min grain), daily, weekly declared
  per series; **custom periods** declared as `{name, period, fourier_order}`
  (for example a 6 h duty cycle). **Monthly and yearly are refused** by the
  spec loader until the series has ≥ 2 full cycles — a hard check, not a warning.
- **Events / holidays**: custom event calendar (deploys, AC on/off, the 34 °C
  incident day) as Prophet holidays and as dummy regressors for the others.
- **Changepoints / regime**: Prophet changepoints, plus a declared
  `train_from` per series so a known break (AC install 2026-09-26) can cut old data.
- **Transforms**: log / Box-Cox for positive series; logit for bounded ones
  (humidity 0–100 %); inverse-transformed before storage.
- **Probabilistic output**: every forecast stores quantiles (0.05, 0.1, 0.25,
  0.5, 0.75, 0.9, 0.95). Point-only forecasts are not a valid result.
- **Conformal calibration**: wrap every model in split-conformal intervals
  computed from backtest residuals per horizon step. This makes "80 % interval"
  mean 80 %, whatever the model's own assumptions say. This is the single most
  important feature after the baseline gate.
- **Step-out forecasting**: recursive for ARIMA-family, direct (one model per
  horizon bucket) for LightGBM; the horizon step is a column, so the scorer can
  report error by lead time.
- **Covariate futures**: each covariate is marked `known_future` (calendar, a
  scheduled job) or `forecast_needed` (chassis watts). `forecast_needed`
  covariates are themselves forecast, or supplied by a scenario. This is the
  bridge to "what if".
- **Ensembles**: median ensemble of models that passed the gate; weights only
  after enough live scores exist.
- **Exceedance probability**: `P(y ≥ threshold within window)` from the
  quantile path — the output hardware-watch actually needs.
- **Anomaly flag**: a reading outside the calibrated 95 % interval is an
  anomaly, published as an event. This replaces nothing yet; it is scored like
  any other output.

## Evaluation harnesses

Four lanes, all writing to the same score tables.

1. **Backtest gate (offline, per fit).** Rolling-origin cross-validation over the
   last N windows (`statsforecast.cross_validation`). Metrics per horizon bucket:
   MAE, RMSE, **MASE against seasonal naive** (< 1 means it beats the baseline),
   pinball loss / CRPS for quantiles, interval coverage and width. Promotion rule:
   a model becomes the champion for a series only when MASE < 1 at the target
   horizons *and* 80 % coverage is within [0.72, 0.88]. Otherwise the
   seasonal-naive baseline is served and the result says so (`served_model=baseline`).
2. **Live scoring reducer.** When a real value arrives for a stored forecast step,
   write a `prediction_score` row (error, quantile hit, coverage hit). Rolling
   7-day live metrics per model × horizon. Cursor stored, so a stalled scorer is
   visible.
3. **Monitoring.** Alerts (bus event + Hub panel) for: live coverage drifting out of
   band; live MASE ≥ 1 for 24 h; forecast staleness (no new forecast in 2× cadence);
   input staleness (sensor dropped out); degenerate output (flat forecast,
   zero-width interval, all-null quantiles). Each alert has a test that makes it fire.
4. **CI eval (deterministic, fast).** A frozen 14-day cabinet fixture (parquet,
   small) plus synthetic series with known seasonality and a known injected
   effect. Asserts: MSTL recovers the injected daily period; conformal coverage
   on the synthetic set lands in band; the spec loader refuses monthly seasonality
   on 48 days; every model beats a deliberately bad baseline. Runtime target < 30 s
   (Prophet fits are the slow part; fixture is sized for it).

Performance lane (inside 1 and 4): fit time, predict time, peak memory per model,
recorded per run. A model that breaks the refit budget is dropped from the
schedule, not the scoreboard.

## "What if" — from forecast to cause and effect (phase 3)

Three levels. Each level is a different claim and must be labelled as such in
the output (`claim_kind`).

1. **Conditional forecast** (`claim_kind=conditional`, phase 2). "If chassis watts
   follow this path, cabinet temp is forecast to follow that one." Runs SARIMAX /
   LightGBM with a supplied covariate path. It is a correlation-based answer. It
   is labelled so.
2. **Past-intervention effect** (`claim_kind=observational_effect`). An action log
   (`prediction_intervention`: who, what, start, end) plus interrupted-time-series
   analysis: forecast the counterfactual from data before the action, compare with
   what happened. Same method as CausalImpact. Weak when the action coincides with
   other changes (AC switching, time of day).
3. **Experimental effect** (`claim_kind=experimental_effect`). Orion runs
   **switchback experiments**: in safe windows, alternate a named workload ON/OFF
   in randomized blocks (for example 20 min each, ≥ 8 blocks), log every block,
   estimate the average temperature difference with a confidence interval. This
   is the only level that can honestly answer "focus on X raises temp by Y ± Z".
   Safety: the existing 34 °C reflex stays the hard stop; experiments start only
   when `P(temp ≥ 32 in 30 min) < 5 %`; quiet hours apply.

The Orion-facing tool (the "grain") is then: `estimate_effect(action, duration)`
→ returns the best available level, its effect size, interval, and `claim_kind`,
and `propose_experiment(action)` when only level 1 exists. Phase 3 dispatch goes
through `orion-self-experiments` and must produce a real dispatched run, not a
validated plan.

Proposal-mode fields for phase 3:
- capability change: Orion can switch a named workload on/off on a schedule to learn its effect.
- data touched: biometrics, cooling samples, new intervention + experiment tables.
- privacy: none beyond existing hardware telemetry.
- trace: `prediction_intervention` rows with correlation IDs, matched GPU-lane events, the effect estimate row.
- dangerous failure: an experiment holds a heat-producing load during an AC failure. Mitigated by the forecast gate plus the unchanged 34 °C reflex.
- disable: `PREDICTION_EXPERIMENTS_ENABLED=false`; experiments end at the current block.

## Proposed schema / API changes

Service: `services/orion-prediction/` (FastAPI + scheduler worker), hosted on
athena (the DB and the sensors live there; CPU only for phases 1–3).

Series registry `config/prediction/series.v1.yaml` — every entry needs a consumer:

```yaml
- series_id: athena.cabinet_temp_c
  source: {table: orion_biometrics_summary, node: athena, json_path: measurements.cabinet_temp_c, ts_column: timestamp}
  grain: 5min
  agg: mean
  gap_policy: {max_fill: 10min}
  transform: none
  seasonalities: [{name: daily, period: 1d, fourier_order: 6}, {name: weekly, period: 7d, fourier_order: 3}]
  covariates: []            # phase 2: chassis_watts (forecast_needed), cooling_watts (forecast_needed), hour (known_future)
  train_from: 2026-09-26
  horizons: [0, 15min, 1h, 6h, 24h]
  thresholds: [{name: shed, value: 34.0, window: 30min}]
  consumers: [hardware-watch, hub-prediction-panel]
```

Postgres (one migration, `-- ORION-MIGRATION-REQUIRED-BY: orion-prediction`):
`prediction_model_run` (series, model, params, data hash, train range, fit
seconds, backtest metrics, champion flag), `prediction_forecast` (run, issued_at,
target_ts, horizon_step, point, quantiles jsonb), `prediction_score` (forecast
id, actual, error, quantile hits), `prediction_intervention` (phase 3),
`prediction_effect_estimate` (phase 3).

Bus (registered in `orion/bus/channels.yaml` + `orion/schemas/registry.py`):
- `orion:prediction:forecast` — `prediction.forecast.v1` (latest champion forecast per series)
- `orion:prediction:exceedance` — `prediction.exceedance.v1` (threshold probability)
- `orion:prediction:alert` — `prediction.alert.v1` (monitoring lane)
- `orion:prediction:anomaly` — `prediction.anomaly.v1`

HTTP:
- `GET /health`, `GET /series`, `GET /series/{id}/models` (scoreboard)
- `GET /series/{id}/forecast?horizon=` (latest stored, no refit)
- `POST /forecast` (ad-hoc: series, model or `champion`, horizon, quantiles)
- `POST /scenario` (phase 2: covariate paths → conditional forecast)
- `POST /backtest` (series, models, windows → metrics)
- `GET /effects/{action}` (phase 3)

## Files likely to touch

- `services/orion-prediction/**` (new: app, settings, Dockerfile, compose, `.env_example`, tests, evals, README)
- `config/prediction/series.v1.yaml` (new)
- `orion/schemas/prediction.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `scripts/sql/manual_migration_<date>_prediction.sql`
- phase 2+: `services/orion-hardware-watch/app/watcher.py` (consume exceedance), Hub panel
- phase 3: `services/orion-self-experiments/**`, intervention logging at the workload's lane

## Non-goals

- No monthly/yearly seasonality until the data supports it.
- No deep-learning training in phases 1–3.
- No change to the 34 °C reflex. Forecasts add an earlier warning; they never remove the hard stop.
- No forecasts fed into cognition/attention until live scoring has 7 days of
  coverage evidence.
- No generic AutoML search. The model list is fixed and small; the gate decides.

## Acceptance checks

Phase 1:
- `pytest services/orion-prediction/tests -q` and the CI eval lane pass.
- Backtest on `athena.cabinet_temp_c` stored in `prediction_model_run`; the champion has MASE < 1 at 1 h and 6 h, or the baseline is served and labelled.
- Live: `prediction_forecast` rows every 5 min; `prediction_score` cursor advancing; after 7 days, live 80 % coverage in [0.72, 0.88].
- Spec loader refuses `monthly` on this series (test).

Phase 2: conditional forecast with chassis watts beats univariate on backtest, or is reported as no gain; hardware-watch logs the exceedance probability next to each reflex decision.

Phase 3: at least one dispatched switchback experiment with ≥ 8 logged blocks and an effect estimate with an interval.

## Recommended next patch

Phase 1 only: service skeleton, series registry with the one cabinet-temp entry,
baselines + AutoETS + AutoARIMA + MSTL + Prophet, conformal intervals,
backtest gate, forecast + score tables, live scoring reducer, `/health` and
`/series/{id}/forecast`, CI eval with the frozen fixture. No consumers wired
yet beyond a Hub-visible scoreboard.
