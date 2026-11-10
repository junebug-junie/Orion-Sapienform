## Summary

- The forecaster's ranges 5 and 15 min ahead now fit the hour. Calm hours get narrow ranges and busy hours get wide ones. Before, every hour got one width.
- How: the range is calibrated on the last 3 days of live scores from the model that serves that horizon now. Each miss is divided by how big the 5-min misses were in the hour before it. The range is then multiplied by the current hour's value (normalized split conformal, Lei et al. 2018).
- These keep the old backtest range: ranges past 15 min, thin pools, a stale last hour, steps served by the baseline after a model failure, and any error in this code. Every stored forecast row records where its range came from (`prediction_forecast.interval_source`, an additive migration).
- Scoring now runs before issuing in each tick, so the newest outcome counts.
- Kill switch: `PREDICTION_ADAPTIVE_INTERVALS_ENABLED` (on by default).

Spec: `docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md` (PR #3168), NEXT item 1 of the 11-10 handoff. Juniper approved it 2026-11-10.

## Outcome moved

The surprise calm check (PR #3170) failed on live data. On a quiet day, the surprise read far below calm, so nothing could consume it. Replayed with this code over 14 days (`evals/run_adaptive_interval_replay_eval.py`), on the same days and the same forecasts for both, days that pass the calm check:

| series | horizon | backtest ranges (served) | adaptive ranges |
|---|---|---|---|
| cabinet | 5 min | 2/13 | 13/13 |
| cabinet | 15 min | 7/13 | 12/13 |
| office | 5 min | 1/6 | 6/6 |
| office | 15 min | 2/6 | 6/6 |

80 % coverage: 0.76-0.81 adaptive vs 0.85-0.93 served. The served ranges were too wide.

The method and the 15-min cut were chosen on these same 14 days, so this replay is not a holdout. The live check after deploy is the holdout: `evals/run_surprise_calm_live_eval.py --hours 24 --source adaptive`.

Live proof is UNVERIFIED until it is deployed and a quiet 24 h has passed: `evals/run_surprise_calm_live_eval.py --hours 24`.

## Current architecture

Each horizon's range was `point + quantile of backtest residuals`, from 40 backtest windows 3 h apart. That is one width for all hours of the day, refreshed every 6 h.

## Why this method, and what failed first

The handoff proposed scaling by input volatility. I tested it before building (`vol_experiment.py`, time-ordered 60/40 split of 14 days of live champion rows).

- **Scale by how fast the reading moved** (mean |diff| over the last 1 h or 3 h): this overcorrects. Smooth ramps that the model tracks well read as "busy". Cabinet 15 min, by tercile: calm +0.19, busy -0.44.
- **Scale by how big the forecaster's own misses were in the last hour**: this works. The terciles flatten and coverage lands at 0.81-0.85.
- **The transform is not the cause.** Exact rank p-values vs the 9-level linear CDF differ by only 0.03-0.05.
- **Drift matters too.** One width calibrated on older live scores over-covered later days (0.87-0.94), because the errors shrank over time. A rolling 3-day pool absorbs that.
- **Past 15 min it hurts.** At 6 h, cabinet went from 10/12 calm days to 8/10. At 60 min, once each pool held only one model's misses, the ranges were too narrow: office excess was +0.33 and coverage 0.73, while the backtest range was already close to calm there (+0.09). So horizons past 15 min keep the backtest range.

## Architecture touched

orion-prediction only. No bus or schema change. One additive migration adds the nullable `prediction_forecast.interval_source` column. It is declared REQUIRED-BY orion-prediction, so the post-pull rebuild applies it before the service rebuilds.

## Files changed

- `services/orion-prediction/app/adaptive.py`: new. Difficulty series and the normalized offsets per horizon.
- `services/orion-prediction/app/engine.py`: `adapt_intervals` on the champion role, `interval_source` state and log line, score-before-issue order.
- `services/orion-prediction/app/store.py`: new `champion_pool` (scored champion rows plus the owning model, joined through `run_id` to the run's assignment). `insert_issue` writes `interval_source`, and the startup table probe checks for it.
- `services/orion-sql-db/manual_migration_prediction_forecast_interval_source_v1.sql`: new additive column.
- `services/orion-prediction/app/effects.py`: the experiment guard skips adaptation, because it only reads exceedance.
- `services/orion-prediction/app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: the kill switch, and `interval_source` in `/series/{id}`.
- `services/orion-prediction/tests/test_adaptive.py`: new tests.
- `services/orion-prediction/evals/run_adaptive_interval_replay_eval.py`: new replay eval.
- `services/orion-prediction/evals/run_surprise_calm_live_eval.py`, `README.md`: docs.

## Schema / bus / API changes

- Added: the `prediction_forecast.interval_source` column, and `interval_source` per series in `/health`.
- Removed / renamed: none.
- Behavior changed: the champion's stored and published quantiles at horizons <= 60 min. The live coverage alert and the surprise gate read these quantiles.
- Compatibility notes: `PredictionForecastV1` is unchanged.

## Env/config changes

- Added keys: `PREDICTION_ADAPTIVE_INTERVALS_ENABLED=true`
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (the primary checkout's `services/orion-prediction/.env` has the key)
- skipped keys requiring operator action: none

## Tests run

```text
orion-prediction image: pytest tests -q -> 314 passed
scripts/check_migration_required_by.py -> OK
git diff --check -> clean
scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
evals/run_adaptive_interval_replay_eval.py --days 14 -> table above (~6 min, read-only)
Pre-build experiments (scratchpad, read-only): miss-size scaling vs reading-speed scaling vs one width; rank p vs 9-level transform
```

## Docker/build/smoke checks

```text
Tests ran inside the orion-prediction-prediction image. Not deployed.
```

## Review findings fixed

The code review (a subagent, high effort, with mutation tests) found no leakage and no timezone bugs. Fixed:

- Finding (medium-high): the pool mixed the misses of different models across reselections. Office 60 min: direct_ar MAE 0.48 vs mstl_arima 0.26, and the 60-min model switched 8 times in 4 days.
  - Fix: each horizon's pool is filtered to the model serving it now (`champion_pool` joins `run_id` to the run's assignment). Rows a failed model handed to the baseline are excluded.
  - Evidence: `test_pool_keeps_only_the_model_serving_the_horizon_now`, `test_rows_a_failed_model_handed_to_the_baseline_are_excluded`. The re-run replay then showed 60 min going too narrow, which led to the 15-min cap.
- Finding (medium): scoring before issuing meant a scoring fault would skip the forecast.
  - Fix: the error is caught and raised as a `score_failed` alert, and the forecast is still issued.
  - Evidence: `test_a_scoring_fault_still_issues_and_alerts`.
- Finding (medium): an error in the adaptive code would cost the whole forecast and the experiment guard.
  - Fix: on any error the forecast keeps the backtest range, with source `backtest:error:<Type>`. The guard calls `build_forecast(adapt=False)`.
  - Evidence: `test_an_adaptive_fault_keeps_the_backtest_range`.
- Finding (medium): nothing recorded which range a stored forecast used.
  - Fix: the `interval_source` column, plus `--source adaptive` on the calm eval. The README endpoint reference is corrected, and the in-memory source is cleared on a stale-input skip.
  - Evidence: the engine test asserts each row's source.
- Finding (medium-low): the tests still passed with score-after-issue, with leakage, or with the fallback ignored (confirmed by mutation).
  - Fix: `test_tick_scores_before_it_issues`, `test_a_rows_difficulty_uses_only_misses_known_at_its_origin`, `test_a_step_served_by_the_baseline_after_a_failure_keeps_the_backtest_range`.
- Finding (low): the replay compared different day sets.
  - Fix: both sets are now scored on the same days and the same forecasts. The in-sample caveat is stated.
- Finding (low): a flat or quantized hour could blow the range up to about 1e8.
  - Fix: the floor is now at least 0.1 x the median difficulty.
  - Evidence: `test_a_flat_hour_cannot_blow_the_range_up`.
- Finding (low): the query cost. Fix: a dedicated query limited to champion rows at horizons of 15 min or less.

## Restart required

Deploy = merge, then `git pull --ff-only` in the primary main checkout. The post-pull hook applies the additive migration, then rebuilds orion-prediction. Live check after a quiet 24 h:

```bash
docker run --rm --network <prediction net> -e POSTGRES_URI=... -v /mnt/scripts/Orion-Sapienform:/w:ro -w /w/services/orion-prediction --entrypoint python orion-prediction-prediction evals/run_surprise_calm_live_eval.py --hours 24 --source adaptive
```

## Risks / concerns

- Severity: medium. Concern: after a model switch at a horizon, that horizon falls back to backtest ranges until the new model has 288 scored forecasts (about a day). Mitigation: this is intended, because a fresh model has no live record. The fallback is visible in `interval_source`.
- Severity: low. Concern: difficulty comes from the 5-min owner's misses, whatever model serves 15 min. Mitigation: it measures how hard the hour is, and it is in the same units because both horizons forecast the same reading.
- Severity: low. Concern: the first hour after a regime switch lags (the difficulty window still holds the old regime). The test measures this. Mitigation: none yet. A shorter window would trade it for noise.
- Severity: low. Concern: exceedance probabilities still use the backtest paths, so they can disagree with the new ranges. Mitigation: they are graded separately (Brier score), and the README documents it.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
