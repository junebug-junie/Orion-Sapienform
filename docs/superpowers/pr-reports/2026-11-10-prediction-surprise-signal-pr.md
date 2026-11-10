## Summary

- orion-prediction now turns each scored cabinet/office forecast into **surprise**: how unlikely the real value was under the forecast, in one unit (`-ln p`), with a calm point set by math (0.95), not by tuning.
- Published as `prediction.surprise.v1` on `orion:prediction:surprise`, only for forecast distances that pass the gate (real forecaster, live 80 % coverage in band, not worse than the baseline live). A bucket with no reading goes out as `outcome=missing`: silence is news.
- `GET /series/{id}/surprise` and `evals/run_surprise_calm_live_eval.py` run the spec's calm check against live data.
- **The calm check fails today.** No consumer is wired; nothing in Orion reads this signal yet.
- Second build patch of the prediction-error suite rebuild spec (PR #3168): "expose orion-prediction's existing scored forecasts as `prediction.surprise.v1` (no new forecasting), which proves the definition end to end."

## Outcome moved

Orion's first prediction-error signal that is a real prediction: forecast stored before the target time, scored when the clock says, surprise in one unit, calm defined by theory.

It also produced the first honest measurement of whether the forecaster's ranges fit calm and busy hours alike. They do not (below).

## Metric quality gate

1. **Provenance.** `app/surprise.py::conformal_p_value` over `prediction_forecast.quantiles` (written by `Engine.build_forecast`, split-conformal from backtest residual paths) and `actual` (written by `Engine.score` from the 5-minute bucket mean).
2. **Independence.** Not wired into any model with other metrics. It shares its input with `in_80`/`in_95` coverage (same rows); coverage is the drift reading, surprise the spike reading, by design.
3. **Theory anchor.** Probability integral transform: under a calibrated forecaster the CDF at the outcome is uniform, so `p = 2 min(u, 1-u)` is uniform and `E[-ln max(p, 0.05)] = 1 - 0.05`. Method bias measured: 9-level linear CDF reads -0.033 for a perfectly calibrated normal (`tests/test_surprise.py`).
4. **Live sanity (2026-11-10, champion rows).** Not degenerate: varies, reaches the floor, returns to rest. But it rests **below** the calm point:

```text
7 days     cabinet 15 min excess -0.115 +-0.056   office 15 min -0.116 +-0.061   (80% coverage 0.84 / 0.85)
quiet 24 h cabinet 15 min -0.281 +-0.101, 60 min -0.372 +-0.145; office 15 min -0.355 +-0.085
```

   The ranges are calibrated on average but have one width for calm and busy hours. A calm day reads too calm and the misses cluster in busy periods. The fix is in the forecaster (volatility-adaptive conformal ranges), not in this transform. Until the calm check passes, **no consumer may read this signal**.
5. **Existing mechanism.** orion-prediction already scores coverage; no surprise or p-value existed anywhere in the repo for these forecasts.
6. **Reversibility.** No table, no migration, no consumer. Remove = drop the module, the channel row and the schema.

## Current architecture

orion-prediction forecasts `athena.cabinet_temp_c` and `office.temp_c` every 5 min at 5 min / 15 min / 1 h / 6 h / 24 h, stores 9 quantiles per horizon, scores each when its bucket lands (`actual`, `in_80`, `in_95`, or `missing`), and monitors live coverage per horizon. Nothing turned a scored miss into a surprise number, and no prediction-error node used any of it.

## Architecture touched

- `services/orion-prediction/app/surprise.py` (new): p-value, surprisal, calm point, excess, per-horizon gate, calm summary.
- `app/engine.py`: monitor computes the gate per horizon; score collects champion outcomes on gated horizons; tick publishes them; `surprise_summary` for the endpoint.
- `app/store.py`: `due_for_scoring` also returns `role, horizon_min, issued_at` and `served_model` (joined from the issuing run's per-horizon assignment); new read `scored_with_quantiles`.
- `app/main.py`: `GET /series/{id}/surprise?hours=`.
- Contract: `PredictionSurpriseV1` (`orion/schemas/prediction.py`), registry, `orion:prediction:surprise` in `channels.yaml`.

## Files changed

- `services/orion-prediction/app/{surprise,engine,store,main}.py`: as above.
- `services/orion-prediction/tests/{test_surprise,test_engine,test_schemas}.py`: math, gate, publish path, missing outcome, contract.
- `services/orion-prediction/evals/run_surprise_calm_live_eval.py`: live calm check (read-only).
- `services/orion-prediction/README.md`: Surprise section, contract line, eval command.
- `orion/schemas/{prediction,registry}.py`, `orion/bus/channels.yaml`: contract.
- `.github/workflows/orion-prediction-tests.yml`: also triggers on registry/channels changes.

## Schema / bus / API changes

- Added: `PredictionSurpriseV1`, kind `prediction.surprise.v1`, channel `orion:prediction:surprise` (producer orion-prediction, no consumer), `GET /series/{id}/surprise`.
- Removed / Renamed: none.
- Behavior changed: none for existing consumers (forecast payload unchanged).
- Compatibility notes: new event, no existing reader.

## Env/config changes

- Added / Removed / Renamed keys: none. Publishing follows the existing `PREDICTION_PUBLISH_ENABLED`.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
docker run orion-prediction-prediction (worktree mounted :ro) pytest tests -> 307 passed
Live DB, READ ONLY transaction: new due_for_scoring join returns served_model per row (e.g. champion|5|direct_arx_overlap)
Mutation: surprise_events without the gate -> test_gated_horizon_publishes_its_outcome_champion_only[True/False] fail
python -c "resolve('PredictionSurpriseV1')" -> PredictionSurpriseV1
```

## Evals run

```text
evals/run_surprise_calm_live_eval.py --hours 168 / --hours 24 (live, read-only): numbers above. Calm check FAILS today.
```

## Docker/build/smoke checks

```text
UNVERIFIED live until deploy. After deploy:
curl -fsS 'http://localhost:8132/series/athena.cabinet_temp_c/surprise?hours=24'   # gate per horizon, excess vs band
docker logs --since 30m <orion-prediction container> 2>&1 | grep -i surprise          # no publish errors
```

## Review findings fixed

- Finding (medium): the gate judged the model serving at scoring time, not the one that issued the forecast (a 6 h forecast could publish as gated after selection changed).
  - Fix: `due_for_scoring` joins `prediction_model_run.assignment` for the row's `run_id`; `surprise_events` gates on that model plus live coverage and its live verdict.
  - Evidence: `test_gate_judges_the_model_that_issued_the_forecast_not_todays`; live read-only query returns `served_model` per row.
- Finding (low-medium): outcomes scored before the first monitor pass after a restart were lost for good.
  - Fix: `score()` computes live metrics inline when none exist yet. Evidence: `test_first_score_after_start_is_judged_not_lost`.
- Finding (low): an exception building events skipped exceedance and shadow scoring for the tick.
  - Fix: guarded and logged (`prediction_surprise_failed`). Evidence: `test_a_surprise_failure_never_costs_the_scores`.
- Finding (low-medium): `calm` read False for a calibrated forecaster over long windows (band below the 0.033 method bias); `mean_excess` and `calm` used different samples; thinning by row index broke on gaps.
  - Fix: thinned mean reported as `mean_excess`; thinning anchored to the clock; `calm` allows `METHOD_BIAS` (0.033, measured). Evidence: `test_long_windows_do_not_call_a_calibrated_forecaster_uncalm`, gap case in `test_summary_thins_overlapping_origins_and_flags_calm`.
- Minor: publish failure now skips one event instead of dropping the batch; non-monotone quantiles refused (`ValueError`, logged); weak tests pinned (exact collapsed p = 0.075; summary gate reason computed from the served assignment).

## Restart required

Post-pull rebuild of `orion-prediction` from the clean primary main checkout after merge.

## Risks / concerns

- Severity: medium. Concern: the signal rests below its calm point (ranges too wide on quiet days). Mitigation: no consumer; the README and the eval say so; next patch = volatility-adaptive ranges in the forecaster, then the substrate node.
- Severity: low. Concern: an event lost on a bus error is not retried. Mitigation: the scored row in `prediction_forecast` is the record; `/surprise` recomputes from it.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
