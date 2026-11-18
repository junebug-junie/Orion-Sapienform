## Summary

- Every contender model now forecasts in the shadow at every issue, whether it serves or not. A contender is any model that beats the baseline somewhere in the backtest, plus any model that serves. The rows go into `prediction_shadow_forecast` and are scored against real readings, the same way served forecasts are.
- `app/live.py` turns those scores into a live verdict per model and horizon:
  - It pairs each shadow forecast with the baseline forecast for the same origin.
  - It keeps one origin per horizon-long block, so the samples are independent, and needs at least 20.
  - It tests both directions, so the verdict is `better`, `worse`, `unclear` or `pending`.
  - `better` is Holm-corrected across every model × horizon tested.
- The gate now uses those verdicts:
  - A live-`worse` model never serves.
  - A live-`better` model serves if it also beats the baseline in the backtest.
  - Otherwise the backtest decides, with hysteresis: p < 0.05 to start serving, p < 0.10 to keep serving.
  - Live winners rank ahead of backtest-only winners.
- `worse_than_baseline` now judges the serving model on its own live record.
- `GATE_VERSION` is now 3. On deploy the old selection is refused and the service reselects.

## Outcome moved

**The flicker.** On 10-21 and 10-22, `direct_arx` at 15 min sat right at p 0.041–0.070. It served for 18 h, during which it beat the baseline live (0.922× over 213 forecasts), then dropped out. With the hysteresis band and a live record, a model with a real but small edge keeps its horizon. A backtest-only win that fails live stops serving.

**The stale alert.** It was firing on the 10-17/18 pick, which had already been dropped. It now judges only the model that is serving.

## Current architecture

- The gate used the backtest only, with a hard p < 0.05 cut-off.
- Live scoring existed only for the served champion and the baseline.
- The alert used 7 days of champion rows.

## Architecture touched

- `services/orion-prediction`:
  - `app/live.py` (new)
  - `select_champion` (`live`/`incumbent` arguments, `STAY_ALPHA`, tiered ranking)
  - engine: contenders, `build_shadow` after the served issue, `score_shadow`, cached per-horizon verdicts, retention, alert
  - store: `insert_shadow`, `due_shadow`, `write_shadow_scores`, `shadow_pairs`, `prune_shadow`
  - settings, `/health`, `/live`
- New migration.

## Files changed

- `services/orion-sql-db/manual_migration_prediction_shadow_v1.sql`: the table and two partial `CONCURRENTLY` indexes. Auto-apply clean.
- `services/orion-prediction/app/{live,backtest,engine,store,settings,main}.py`.
- `services/orion-prediction/{.env_example,docker-compose.yml,README.md}`: `PREDICTION_SHADOW_ENABLED=true`.
- `services/orion-prediction/tests/test_live.py` (new), `tests/test_engine.py`. The two old alert tests were replaced, because they checked the champion-row history that the alert no longer uses.
- The spec.

## Schema / bus / API changes

- Added:
  - table `prediction_shadow_forecast`
  - `/live` → `shadow` (verdicts per `model@horizon`)
  - `/health` → `shadow`, `shadow_failed`
  - per-model metrics `live`, `passed_by`
- Removed:
  - `/live` → `losing_horizons`
  - the champion-row alert logic
- Renamed: none.
- Behavior changed: see the gate rules in the summary. The alert judges the serving model. `GATE_VERSION` is now 3.
- Compatibility notes:
  - Deploy forces one reselect, about 5 min.
  - Verdicts start as `pending`. The first possible verdicts come after about 100 min (5 min horizon), about 5 days (6 h) and about 25 days (24 h).

## Env/config changes

- Added keys: `PREDICTION_SHADOW_ENABLED=true`. Setting it to false also turns off `worse_than_baseline`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q -> 133 passed
migration: auto_apply_refusals -> [], REQUIRED-BY OK
```

## Evals run

```text
null calibration (test_live.py): Holm keeps any-'better' across 4 models x 3 horizons <= 10% of 60 trials
  (uncorrected: ~1/3 of selections, measured by the reviewer)
live smoke (branch image, real telemetry read-only, throwaway DB): 4 contenders (direct_ar, direct_arx,
  mstl_arima, window_average_1h), 20 shadow rows per issue, ~13 s CPU per issue, no failures
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-prediction build -> built; smoke above
```

## Review findings fixed

- Finding: the 24 h horizon could never reach a verdict (7 days hold at most 7 independent forecasts).
  - Fix: a per-horizon look-back of max(7 d, 25 horizons), and 35-day retention.
  - Evidence: `test_a_24h_horizon_reaches_a_verdict_inside_its_own_window`.
- Finding: running many tests let a no-better model serve by luck in about 1/3 of selections, skipping the backtest.
  - Fix: Holm correction across the family, and live can only promote a model that also wins the backtest.
  - Evidence: `test_holm_keeps_null_better_verdicts_rare_across_many_tests`, and the "live can't overrule a backtest loss" case in the gate test.
- Finding: live and backtest scores were ranked on mixed scales.
  - Fix: tiered ranking.
  - Evidence: `test_a_live_winner_outranks_a_better_looking_backtest_only_model`.
- Finding: `shadow_pairs` could not use its partial index, and it ran every tick.
  - Fix: filter on `scored_at`, and cache the verdicts per bucket.
- Finding: the alert crashed when the live ratio was None.
  - Fix: guarded.
  - Evidence: `test_alert_formats_a_worse_verdict_without_a_ratio`.
- Finding: non-finite shadow forecasts poisoned the verdicts.
  - Fix: they are skipped when building rows and ignored in the verdicts.
  - Evidence: `test_nan_errors_are_ignored`.
- Finding: shadow fitting delayed the served forecast, and a shadow-table failure cost the served forecast.
  - Fix: the shadow models fit after the served issue is stored, in their own transaction.
  - Evidence: `test_a_shadow_store_failure_never_costs_the_served_forecast`.
- Finding: shadow failures were invisible.
  - Fix: logged, and exposed as `shadow_failed` in `/health`.
- Finding: verdicts flickered as the window slid.
  - Fix: thinning is anchored to absolute horizon blocks.
  - Evidence: `test_thinning_is_anchored_so_a_sliding_window_keeps_the_same_samples`.
- Finding: turning shadow off silently removed the alert.
  - Fix: documented in `.env_example` and the README.

## Restart required

From the clean primary `main` checkout after merge. The post-pull hook applies the migration first, then:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: about 13 s of CPU per 5-minute issue (4 contenders).
  - Mitigation: it runs after the served forecast is published; `PREDICTION_SHADOW_ENABLED` is the switch.
- Severity: low.
  - Concern: a live-`better` model can keep serving for days after a regime change, until its record falls below significance.
  - Mitigation: it must also keep beating the baseline in the backtest, which is re-checked every 6 h.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2844

🤖 Generated with [Claude Code](https://claude.com/claude-code)
