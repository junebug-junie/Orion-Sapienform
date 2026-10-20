## Summary

- New table `prediction_change_event`. It records known moments the world changed under a series. Two effects:
  - `exclude`: masks a window out of training, actuals, covariates and the cause-and-effect paths.
  - `break`: a changepoint. Prophet fits a level shift there once a day of data exists after it. The break becomes every model's training start once enough data exists for the backtest (10 days at the default settings).
- Seeded from the athena biometrics history:
  - **T10 swap**, break at 2026-10-15 18:57 UTC.
  - **Vision bring-up with the cabinet open**, excluded 2026-10-16 21:30 → 2026-10-17 00:35 UTC.
  - **Cabinet sensor geometry changed**, break at 2026-10-17 00:35 UTC.
- API:
  - `POST /change-events`: record an event. Start times in the future are refused.
  - `DELETE /change-events/{id}`: retract an event (kept in the table, ignored from then on).
  - `GET /change-events`: list events.
- `/health` shows each series' training start, the break that set it, and pending breaks with the date each one adopts.
- Each selection records a fingerprint of the events it was built with. Any change to the applied events triggers one reselect.

## Outcome moved

Before this, the forecaster trained across both cabinet regime changes as if they were one regime. Measured: one day after a 2 °C step, Prophet's error is 0.009 °C with the break input and 0.93 °C without it.

What the history shows:

| change | evidence |
|---|---|
| T10 swap | GPU0 temperature 70 → 55 °C; fan resting speed ~44 → ~65 %; idle chassis power ~385 → ~425 W |
| vision bring-up | lidar 116 → 94 mm and magnetic 36 → 33.5 µT (the sensor's surroundings changed); the cabinet reads about 2 °C higher afterwards |

Since 2026-10-18 the cabinet has crossed 34 °C on 278 readings, with the AC already at its ~860 W ceiling.

## Current architecture

The only manual changepoint was the registry's static `train_from` / `exclude`. Nothing recorded hardware or physical changes.

## Architecture touched

`services/orion-prediction`:

- `app/events.py` (new)
- engine: `effective()`, `change_events()` fallback, fingerprinted `selection_due` with backoff, covariate masking
- Prophet break regressor
- effects paths: exclusion masking, break refusal, clipped experiment profile
- store
- API

One new migration. No bus or env changes.

## Files changed

- `services/orion-sql-db/manual_migration_prediction_change_events_v1.sql`: the table (with `retracted_at`) and three seeded events. Auto-apply allowlist: no refusals.
- `services/orion-prediction/app/events.py`:
  - `with_events` (folds events into the spec)
  - `adopt_after` (how old a break must be before it is adopted)
  - `breaks_between`
- `app/engine.py`, `app/models.py`, `app/effects.py`, `app/store.py`, `app/main.py`, `app/series.py`: see "Architecture touched".
- `tests/test_events.py` (new), `tests/test_engine.py`, `tests/test_effects.py`.
- `services/orion-prediction/README.md`, the spec.

## Schema / bus / API changes

- Added:
  - table `prediction_change_event`
  - endpoints `POST`, `GET`, `DELETE /change-events`
  - `/health` field `change_events`
  - blend metrics `train_from`, `events_key`
  - alert code `change_events_unreadable`
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Training excludes the seeded door-open window.
  - Prophet gets the pending breaks as level-shift inputs.
  - The seeded breaks adopt as the training start on 10-25 18:57 and 10-27 00:35 UTC. Each one triggers a reselect.
- Compatibility notes: the first deploy reselects once, because older selections have no event fingerprint.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q -> 81 passed
scripts/check_migration_required_by.py --base origin/main -> OK
auto_apply_refusals(migration) -> []
migration applied twice to a throwaway Postgres -> 3 rows, idempotent
```

## Evals run

```text
Prophet break regressor, synthetic 2 C step, 1 day after: error 0.009 C with vs 0.93 C without (pinned in test_events.py)
Replay eval unchanged (fixture predates the events)
```

## Docker/build/smoke checks

```text
No dependency change; not rebuilt locally.
```

## Review findings fixed

- Finding: after an event change, a failed reselect re-ran the full backtest on every tick.
  - Fix: the backoff is honoured.
  - Evidence: `test_a_failed_reselect_after_an_event_change_backs_off_instead_of_looping` (1 call in 3 ticks; it was 3).
- Finding: an unreadable events table stopped forecasting for every series.
  - Fix: fall back to the last good list, and raise the `change_events_unreadable` alert.
  - Evidence: `test_an_unreadable_events_table_keeps_forecasting_on_the_last_good_list`.
- Finding: intervention estimates and experiment analysis ignored events and exclusions.
  - Fix: the estimate refuses a history window that crosses a break; exclusions are masked; the experiment's time-of-day profile starts at the last break.
  - Evidence: `test_intervention_estimate_refuses_a_history_window_that_crosses_a_break`.
- Finding: the fixed 10-day adoption age ignored the backtest settings.
  - Fix: `adopt_after()`, derived from the window count, step and horizon, with 10 days as the floor.
  - Evidence: `test_adoption_age_grows_with_the_backtest_it_must_feed`.
- Finding: a new event changed live Prophet output without a reselect.
  - Fix: an event fingerprint on each selection.
  - Evidence: `test_a_new_change_event_forces_one_reselect_then_settles`.
- Finding: the Prophet break regressor could be fitted on a handful of points.
  - Fix: at least one day of data after the break is required.
  - Evidence: `test_prophet_skips_a_break_with_less_than_a_day_after_it`.
- Finding: there was no way to retract an event.
  - Fix: `retracted_at` and `DELETE /change-events/{id}`.
- Finding: future start times were accepted silently.
  - Fix: they are refused (5 min skew allowed).
- Finding: `/health` listed superseded breaks as adopted.
  - Fix: `break_adopted` shows only the break that sets the training start.
- Finding: change events did not reach covariates.
  - Fix: the target's exclusions mask its covariates.

## Restart required

Run from the clean primary `main` checkout after merge. The post-pull hook applies the migration (it is allowlist-clean), then:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: part of the cabinet's ~2 °C rise may be the sensor's new position, not heat. The data cannot separate the two.
  - Mitigation: a physical check of the sensor board. That position also affects the 34 °C reflex.
- Severity: low.
  - Concern: events are entered by hand.
  - Mitigation: a follow-up could auto-propose events from step changes in the cabinet's lidar/magnetic channels and in fan/GPU readings.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
