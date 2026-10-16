## Summary

- Two new forecasting models in orion-prediction:
  - `direct_ar` uses only the cabinet temperature's own recent history.
  - `direct_arx` uses the same history plus chassis power and AC power, as known at forecast time.
  - Each one predicts each lead time directly, so neither ever needs a future power value.
- The pair is a standing test on the scoreboard: `direct_arx` vs `direct_ar` shows what the power inputs actually add.
- The replay eval now passes the real power inputs and uses 40 windows, the same as production. The CI step is updated to match.
- If an input sensor dies, only the forecast steps that depend on it fall back to the baseline, and a `model_fallback` alert fires. The rest of the forecast is still issued.

## Outcome moved

On real cabinet history (40 windows), the served forecast's error drops:

| horizon | before | after | change | served by |
|---|---|---|---|---|
| 5 min | 0.068 °C | 0.053 °C | -22% | `direct_ar` |
| 15 min | 0.164 °C | 0.140 °C | -15% | `direct_arx` |
| 1 h | 0.460 °C | 0.430 °C | -7% | `direct_arx` |

6 h and 24 h are unchanged (`naive`, `mstl_arima`).

The ablation says most of the gain comes from the history features, not from the power inputs:

| inputs | 1 h error vs persistence |
|---|---|
| none | 0.98 |
| AC only | 0.94 |
| both | 0.93 |
| chassis power only | 0.97 (adds almost nothing at this resolution) |

## Current architecture

orion-prediction's model zoo used no covariates. Chassis power and AC power were used only by the `/scenario` conditional forecast.

## Architecture touched

- `services/orion-prediction`:
  - `Forecaster.uses_covariates`
  - covariate slicing per backtest window
  - `Engine.load_covariates` used in select and serve
  - per-model fallback when one model fails
- No bus, schema or env changes.

## Files changed

- `services/orion-prediction/app/direct.py`: new. Direct ridge models fitted per anchor lead, with a covariate freshness check.
- `app/models.py`: the covariate-aware `Forecaster`, and the two new zoo entries.
- `app/backtest.py`: passes covariates into each window.
- `app/engine.py`: covariate loading (real buckets only), per-model fallback, and the covariate list stored with each selection and checked on restore.
- `tests/test_direct.py`, `tests/test_engine.py`: new tests.
- `evals/run_cabinet_replay_eval.py`, `.github/workflows/orion-prediction-tests.yml`: covariates in the replay, 40 windows.
- `README.md` and the spec: model table, and the metric quality gate record.

## Schema / bus / API changes

- Added: none on the bus. Blend run rows carry `metrics.covariates`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Two more models are backtested every selection (about 2 s of extra CPU).
  - New alert code `model_fallback`.
- Compatibility notes: a selection stored before this PR records no covariate list. So after deploy, restore reselects once (about 5 min).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q -> 66 passed
pyflakes app tests evals -> clean (one intentional fixture import)
```

## Evals run

```text
evals/run_cabinet_replay_eval.py (40 windows, real history, covariates) -> PASS
  assignment {5: direct_ar, 15: direct_arx, 60: direct_arx, 360: naive, 1440: mstl_arima}
ablation (scratch, same windows): see Outcome moved
```

## Docker/build/smoke checks

```text
Not rebuilt locally. No new dependency: scikit-learn was evaluated and removed together with the GBM variant.
```

## Review findings fixed

- Finding: a dead covariate made build_forecast raise. No champion and no baseline were issued, for up to 6 h.
  - Fix: each model runs under its own guard. Steps owned by a failed model take the baseline's path and residuals. `model_fallback` alert fires.
  - Evidence: `test_a_dead_covariate_falls_back_to_the_baseline_for_its_steps_only`.
- Finding: a stale covariate was forward-filled and used for up to about 2 days.
  - Fix: `direct_arx` refuses a covariate with no real bucket in the 30 min before the origin.
  - Evidence: same test (covariate dropped for the last hour).
- Finding: the backtest interpolated covariates across the origin.
  - Fix: covariates load unfilled in both backtest and serving. Filling happens only inside each training window. Docstrings reworded.
- Finding: the engine's covariate path had no tests.
  - Fix: two engine tests added (fallback, and restore with different covariates).
- Finding: restore ignored a changed covariate list.
  - Fix: the list is recorded with the selection and compared on restore.
  - Evidence: `test_restore_refuses_a_selection_made_with_different_covariates`.
- Nits:
  - Time of day no longer depends on the timestamp unit.
  - The redundant `x - mean` feature is dropped.

Not fixed, known:
- Each issue re-reads 21 days of raw AC rows (about 2 s of thread time). Follow-up: aggregate in SQL.
- `exclude` windows become interpolated training targets for the direct models. The exclude list is empty today.

## Restart required

Run from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: the gains at 15 min and 1 h are a few percent, measured on 40 windows. They may not hold.
  - Mitigation: the per-horizon gate re-decides every 6 h, and live scoring shows the true rate.
- Severity: low.
  - Concern: AC power is not independent of the temperature (the AC responds to heat).
  - Mitigation: it is used only as a leading indicator, never to claim cause and effect.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2716

🤖 Generated with [Claude Code](https://claude.com/claude-code)
