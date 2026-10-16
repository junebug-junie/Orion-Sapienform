## Summary

- New service `orion-prediction`. It forecasts the Athena cabinet temperature every 5 minutes, at 5 min, 15 min, 1 h, 6 h and 24 h ahead. Every forecast comes with an 80 % and a 95 % range.
- Every forecast is graded when the real reading arrives. The service tracks how often the real value lands inside the ranges, and whether the forecast beats a simple baseline.
- A model is used at a forecast distance only when it beats the better of two baselines ("same as now", "same as yesterday at this time") on past data. The served forecast is stitched from the winners. On real data: persistence @5 min, MSTL @15 min, 1-hour average @1 h, Prophet @6 h and @24 h.
- Cause and effect in three labelled levels:
  - `conditional`: scenario forecasts. This is correlation.
  - `observational_effect`: logged actions, analysed against their own history, with placebo-checked intervals.
  - `experimental_effect`: randomised ON/OFF switchback experiments. Only this level can answer "doing X warms the cabinet by Y".
- hardware-watch shows the forecast's chance of reaching 34 °C beside each reflex decision. This is observe-only: it never sheds, and it never blocks a shed.

## Outcome moved

Before: Orion had no forecasts and no way to grade a prediction against reality. hardware-watch reacted to cabinet heat only after it crossed 34 °C. Now:
- Orion has a stored, scored forecast for every 5-minute bucket.
- There is a live "chance of overheating in 30 min" next to the reflex.
- There is an honest path, with labelled confidence, to "if I do X, how much hotter does it get".

## Current architecture

- Cabinet readings arrive about every 31 s in `orion_biometrics_summary`. The AC plug readings are in `home_cooling_sample` (since 2026-09-26).
- hardware-watch sheds GPU load at ≥ 34 °C.
- No forecasting code existed.
- `orion/substrate/prediction_error_magnitude.py` measures surprise against history, not against a prediction.

## Architecture touched

- New service `services/orion-prediction`: select → issue → score → monitor loop, FastAPI on port 8132, Postgres tables.
- New contract: `orion:prediction:forecast` / `PredictionForecastV1`.
- Consumer added: `services/orion-hardware-watch` subscribes to it (observe-only).
- New config: `config/prediction/series.v1.yaml` (series registry, baked into the image).
- `scripts/sync_local_env_from_example.py`: added `orion-prediction`, `orion-hardware-watch` and the `PREDICTION_` / `HARDWARE_WATCH_` prefixes. Before this, a default sync added nothing for either service.

## Files changed

- `services/orion-prediction/app/series.py`: the series registry loader. It refuses a bad config.
- `services/orion-prediction/app/frame.py`: puts readings on a regular grid. Short gaps are filled; long gaps stay empty.
- `services/orion-prediction/app/models.py`: the models (baselines, MSTL+ETS/ARIMA, Prophet). A seasonality needs 2 full cycles before it is used.
- `services/orion-prediction/app/backtest.py`: the rolling backtest, the per-horizon gate, split-conformal ranges, and the threshold chance from past error paths.
- `services/orion-prediction/app/engine.py`: the loop, the scoring step, monitoring alerts, and restore after a restart.
- `services/orion-prediction/app/causal.py`: SARIMAX scenarios, interrupted time series with placebo calibration, and switchback design and analysis.
- `services/orion-prediction/app/effects.py`: the causal endpoints, the experiment guard, design matching, and the best-effect answer.
- `services/orion-prediction/app/store.py`, `main.py`, `settings.py`: Postgres, HTTP and settings.
- `services/orion-prediction/tests/*` (60 tests), `evals/*`, and the fixture `evals/fixtures/cabinet_5min.csv.gz` (real 5-min means, 2026-09-26 → 2026-10-15).
- `services/orion-prediction/{Dockerfile,docker-compose.yml,requirements.txt,.env_example,README.md}`
- `services/orion-sql-db/manual_migration_prediction_v1.sql`: additive tables.
- `orion/schemas/prediction.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: the bus contract.
- `config/metrics/metric_definitions.lock.json`: re-locked for the new channel.
- `services/orion-hardware-watch/app/{watcher,main,settings}.py`, its `.env_example`, `docker-compose.yml`, and tests: the observe-only consumer.
- `.github/workflows/orion-prediction-tests.yml` (new); `.github/workflows/orion-hardware-watch-tests.yml` (path trigger).
- `scripts/sync_local_env_from_example.py`: the sync coverage gap above.
- `docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md`: the design, plus an implementation-status section.

## Schema / bus / API changes

- Added:
  - Channel `orion:prediction:forecast` (`PredictionForecastV1`, kind `prediction.forecast.v1`). Producer: orion-prediction. Consumer: orion-hardware-watch.
  - HTTP on orion-prediction:
    - `/health`
    - `/series`
    - `/series/{id}/models`
    - `/series/{id}/forecast`
    - `/series/{id}/live`
    - `/scenario`
    - `/interventions`
    - `/effects/estimate`
    - `/experiments/guard`
    - `/experiments/design`
    - `/experiments/{id}/analyze`
    - `/effects/{action}`
  - Tables: `prediction_model_run`, `prediction_forecast` (scores are columns on this table), `prediction_experiment`, `prediction_intervention`, `prediction_effect_estimate`.
  - hardware-watch `/health`: a `forecast` key, plus `reflex_shed.forecast`.
- Removed: none.
- Renamed: none.
- Behavior changed: hardware-watch reflex decisions and the on/off log line now show `forecast_p_shed`. Shed decisions are unchanged (tests prove a low forecast never stops a 34.2 °C shed, and a broken forecast can't break the tick).
- Compatibility notes:
  - `orion/autonomy/self_shed.py` reads only `reflex_shed.reason` and `reflex_shed.active`, so the new key is harmless.
  - The schema requires timezone-aware timestamps.

## Env/config changes

- Added keys:
  - orion-prediction: `PREDICTION_SERIES_PATH`, `PREDICTION_ENABLED=true`, `PREDICTION_PUBLISH_ENABLED=true`, `PREDICTION_TICK_SEC`, `PREDICTION_SETTLE_SEC`, `PREDICTION_SELECT_INTERVAL_SEC`, `PREDICTION_BACKTEST_WINDOWS`, `PREDICTION_BACKTEST_STEP_MIN`, `PREDICTION_PORT`, plus the standard service keys.
  - hardware-watch: `HARDWARE_WATCH_FORECAST_SERIES`, `HARDWARE_WATCH_FORECAST_THRESHOLD`, `HARDWARE_WATCH_FORECAST_STALE_SEC`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, both services. Flags default ON in `.env_example` and in `settings.py`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes.
  - hardware-watch got +3 keys.
  - The sync can't create a new service's `.env`, so `services/orion-prediction/.env` was created from `.env_example`, with `POSTGRES_URI` copied from hardware-watch.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q            -> 60 passed
services/orion-hardware-watch: pytest tests -q        -> 81 passed
services/orion-hardware-watch: evals/run_rules_replay_eval.py -> PASS
static gates (every command in orion-static-gates.yml) -> pass, incl. metric lineage + definition drift (re-locked)
repo-level scanning tests: 13 failures, identical set on main (pre-existing, unrelated)
CI on PR (before review fixes): 17/17 checks pass
```

## Evals run

```text
evals/run_cabinet_replay_eval.py --windows 16 (real cabinet history) -> PASS
  per-horizon MAE °C (rel. to better baseline): naive 0.073 @5m; mstl_ets 0.148 (0.84) @15m;
  window_average_1h 0.463 (0.87) @1h; prophet 0.957 (0.78) @6h, 2.305 (0.89) @24h; monthly refused
  conditional scenario: +100 W chassis goes with +0.79 °C; +100 W AC goes with +1.67 °C (reverse causation:
  the AC works harder when hot -- why scenario output is labelled correlation)
evals/run_causal_calibration_eval.py -> PASS
  switchback 95% coverage 0.97, null false-positive rate 0.05; ITS placebo 95% coverage 0.95 (20/20 with interval)
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-prediction build -> image built (896 MB)
in-image smoke: prophet / mstl_ets / mstl_arima fit + forecast, app.main imports
live-data smoke: real telemetry (SELECT only) + throwaway Postgres for all writes (migration applied twice: idempotent)
  tick 123 s (14 windows); forecast 30.6 °C now -> 29.8 °C @6h (80% 28.5–32.1); P(>=34 in 30 min)=0.0;
  experiment guard P(>=32 in 30 min)=0.07 -> unsafe; restore after restart OK; scoring wrote 8 scores;
  ITS on a non-event: +0.27 °C, CI [-0.83, 1.67], p=0.81 (correctly null). Throwaway container removed.
```

## Review findings fixed

- Finding: a forecast with no timezone made the hardware-watch reflex step raise on every tick. The 34 °C shed would then never be asserted or cleared.
  - Fix: the schema requires `AwareDatetime`; `forecast_view` can never raise; `note_forecast` ignores timestamps it can't compare.
  - Evidence: `test_naive_timestamp_forecast_is_refused_by_the_schema`, `test_a_forecast_view_that_raises_cannot_stop_the_reflex`.
- Finding: the gate compared each model on its own windows. A model that failed the hard windows looked 0.40× when it was really 1.49× worse.
  - Fix: the comparison is paired on shared windows, and a model needs ≥ 90 % of the baseline's windows.
  - Evidence: `test_gate_is_paired_...`, `test_gate_refuses_a_good_model_that_skipped_too_many_windows`.
- Finding: `/experiments/guard` could say "safe" from telemetry up to about 2 days stale.
  - Fix: `build_forecast` refuses stale input, and the guard returns unsafe.
  - Evidence: `test_build_forecast_refuses_stale_input_even_outside_issue`, `test_guard_says_unsafe_when_input_is_stale`.
- Finding: re-estimating one intervention made the pooled interval narrower by √k.
  - Fix: only the newest estimate per intervention is pooled.
  - Evidence: `test_reestimating_one_intervention_does_not_narrow_the_pooled_interval`.
- Finding: the "experimental" label was not checked against the randomised design.
  - Fix: logged blocks must match design blocks (same arm, start/end within 2 min, same action), or analysis is refused. Status becomes analyzed only when there is an effect.
  - Evidence: `test_blocks_that_do_not_follow_the_design_are_refused[3 cases]`.
- Finding (minor): early backtest windows fitted the weekly cycle on fewer than 2 cycles.
  - Fix: seasonalities are decided per window.
  - Evidence: `test_backtest_decides_seasonalities_per_window_and_scores_against_real_actuals`.
- Finding (minor): backtest actuals included interpolated buckets.
  - Fix: actuals are real readings only, with exclusions masked (same test).
- Finding (minor): restore did not check the stored selection against the registry.
  - Fix: restore checks grain, residual width, horizons and model names, and reselects on a mismatch.
  - Evidence: `test_restore_refuses_a_selection_that_no_longer_fits_the_registry`.
- Finding (minor): one series' exception stopped every other series.
  - Fix: each series runs in its own try/except.
  - Evidence: `test_one_series_failing_does_not_stop_the_others`.
- Finding (minor): the guard and selection could race.
  - Fix: an immutable `Served` snapshot, swapped in as one object.
- Finding (minor): a failed select looked like a fresh selection in `/health`.
  - Fix: a separate `next_select_at`.
- Finding (minor): placebo windows were drawn with replacement.
  - Fix: distinct offsets, drawn without replacement; pre-periods must be clear of other actions, with each action's span extended by its lag.
  - Evidence: causal calibration eval still PASS.
- Finding (minor): the API accepted timestamps with no timezone and non-UUID ids.
  - Fix: `AwareDatetime` and `uuid.UUID` field types.
- Finding (minor): conformal ranges used interpolated quantiles.
  - Fix: order statistics, rounded outward. The docstring states the clip-to-max limit.
- Finding (minor): the shed probability is on the 5-min mean, not a raw reading.
  - Fix: documented in the schema, the registry and the spec. A max-aggregated series is the follow-up if the number is ever used to act.
- Nits:
  - Stale `orion:prediction:alert` mentions removed.
  - The registry consumer list now names hardware-watch.
  - Distinct `issue_failed` / `degenerate_forecast` alert codes.
  - Blend `n_windows` counts windows with results.
  - The stale cutoff includes settle.

## Restart required

Each line is one command:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_prediction_v1.sql
scripts/safe_docker_build.sh orion-prediction up -d --build
scripts/safe_docker_build.sh orion-hardware-watch up -d --build
```

Run them from the clean primary `main` checkout after merge. Then:

```bash
curl -fsS http://localhost:8132/health
```

The first selection takes about 5–6 min.

## Risks / concerns

- Severity: medium.
  - Concern: there is no experiment dispatcher. Nothing switches a workload on or off. "Focus on X" is still not a named, loggable action.
  - Mitigation: the README and spec say so. Interventions and blocks are logged via HTTP. The guard and the 34 °C reflex stay in place. A dispatcher needs proposal mode, because it is autonomy.
- Severity: low.
  - Concern: only about 20 days of post-AC data. Weekly seasonality is barely usable, and monthly is refused.
  - Mitigation: refusal is automatic and recorded per selection.
- Severity: low.
  - Concern: selection takes about 5–6 min of CPU every 6 h, and issuing pauses during it (about 1 bucket).
  - Mitigation: `forecast_stale` alerts only after 3 buckets plus settle.
- Severity: low.
  - Concern: the "room" goal needs a room sensor. Today's only sensor is in the cabinet.
  - Mitigation: noted in the spec's open questions.
- Live deploy: UNVERIFIED until the migration is applied and the service is up. The live path was proven against real telemetry, with writes going to a throwaway database.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2699

🤖 Generated with [Claude Code](https://claude.com/claude-code)
