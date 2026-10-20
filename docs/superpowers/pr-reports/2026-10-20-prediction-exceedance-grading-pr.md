## Summary

- Every issued "chance of crossing a threshold" (34 °C shed, 32 °C experiment guard) is now stored, in the new table `prediction_exceedance`.
- When the window has passed, each stored chance is scored two ways:
  - `hit_raw`: did any single reading cross the threshold? This is what the cooling reflex reacts to.
  - `hit_mean`: did any 5-minute average cross it? This is what the forecast models.
- `GET /series/{id}/live` grades each threshold: Brier score, base rate, skill against the base rate, and calibration bins.
- The `worse_than_baseline` alert used to fire on a tie. It read "1.00x" while 4 of 5 horizons were served by the baseline itself. It now:
  - judges each horizon separately;
  - uses only forecasts where champion and baseline differ;
  - counts only non-overlapping forecasts, and needs at least 20 of them;
  - fires only when the champion is ≥ 1.05x worse and the loss is significant (p < 0.05).

## Outcome moved

- **The false alarm is gone.** Run read-only against live data, the new metric finds 0 differing forecasts in the last 24 h, so the alert clears.
- **The overheat warning is now measurable.** Before, it was only displayed beside each cooling decision, so its accuracy was unknown. Grading starts at deploy, because past chances were never stored.

## Current architecture

- The exceedance chance was published on the bus and logged by hardware-watch, but never persisted.
- The alert pooled 24 h of pairs across all horizons, in arbitrary order, including identical forecasts.

## Architecture touched

`services/orion-prediction`:

- `app/grading.py` (new)
- engine: issue write, `score_exceedances`, `exceedance_grades`, the per-horizon alert
- store: `insert_issue`, which writes forecast and exceedance rows in one transaction
- series validation
- `/live`

One new migration. No bus or env changes.

## Files changed

- `services/orion-sql-db/manual_migration_prediction_exceedance_v1.sql`: the table and two partial `CONCURRENTLY` indexes. Auto-apply clean.
- `services/orion-prediction/app/grading.py`, `app/engine.py`, `app/store.py`, `app/main.py`, `app/series.py`: see "Architecture touched".
- `tests/test_grading.py` (new), `tests/test_engine.py`, `tests/test_backtest.py`, `tests/test_series.py`.
- `README.md`, spec.

## Schema / bus / API changes

- Added:
  - table `prediction_exceedance`
  - `/live` → `exceedance` (per threshold: `vs_single_reading`, `vs_bucket_mean`)
  - `/live` → `horizons[h]` fields `n_differing`, `n_independent`, `rel_mae_differing`, `p_baseline_better`, plus `losing_horizons`
- Removed: `/live` fields `rel_mae_vs_baseline_24h`, `n_pairs_24h`.
- Renamed: none.
- Behavior changed: `worse_than_baseline` alerts only on a per-horizon, independent, significant loss.
- Compatibility notes:
  - The service refuses to start until the table exists. The post-pull hook applies the migration.
  - Each issue's forecast rows and exceedance rows commit together.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q -> 90 passed
check_migration_required_by -> OK; auto_apply_refusals -> []; migration applied twice to throwaway Postgres -> 4 indexes, idempotent
live read-only smoke (branch image vs production DB): n_differing 0 in 24 h -> alert clears
```

## Evals run

```text
No new eval: grading itself is the live evaluation of the warning; synthetic calibration checks are in test_grading.py
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-prediction build -> built
```

## Review findings fixed

- Finding: the alert's p-value was too optimistic at long horizons. Overlapping forecasts were counted as independent, and the rows were unordered.
  - Fix: judge each horizon separately, keep every (h/grain)-th origin in time order, and require ≥ 20 independent forecasts.
  - Evidence: `test_a_long_horizon_is_not_judged_until_it_has_enough_independent_forecasts`.
- Finding: a failed exceedance insert lost that bucket's rows and its bus publish for good.
  - Fix: one transaction for the whole issue, so a failure means the bucket is re-issued on the next tick.
  - Evidence: `test_a_failed_issue_write_leaves_nothing_and_the_bucket_is_reissued`.
- Finding: a partly observed window was graded as "no crossing".
  - Fix: an incomplete window scores None, unless a crossing was seen. `n_buckets_observed` is stored.
  - Evidence: `test_a_partly_observed_window_is_unknown_unless_a_crossing_was_seen`.
- Finding: the per-tick queries had no indexes.
  - Fix: two partial `CONCURRENTLY` indexes, which the allowlist accepts once the file has no transaction wrapper.
- Finding: the 24 h recency filter excluded the 24 h horizon entirely.
  - Fix: a 7-day window filtered by target time. The 24 h horizon still needs 20 independent days before it is judged, and that limit is stated in `/live`.
- Finding: a threshold window could be longer than the path the chance was computed on.
  - Fix: the registry now refuses it.
- Finding: grades were grouped by name only.
  - Fix: they are also filtered by threshold value and window.
- Finding: `grade()` returned a different shape for empty input, and skill was None for rare events.
  - Fix: the shape is consistent; the docstring explains when skill is None.

## Restart required

From the clean primary `main` checkout after merge. The post-pull hook applies the migration first, then:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: `hit_raw` is a different event from the one the chance models (a single reading vs a 5-minute average), so it will read as under-forecast.
  - Mitigation: both outcomes are graded, and the gap is the measured size of that understatement.
- Severity: low.
  - Concern: the 24 h horizon can't be judged by the alert for about 20 days.
  - Mitigation: stated in `/live` (`n_independent`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2805

🤖 Generated with [Claude Code](https://claude.com/claude-code)
