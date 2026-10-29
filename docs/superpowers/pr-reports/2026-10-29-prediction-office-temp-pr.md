## Summary

- orion-prediction now forecasts the office (room) temperature, `office.temp_c`, as a second target, from the pan-1 SHT40 room sensor Juniper installed 2026-10-28.
- New source kind `office_env` reads `office_env_sample`. The column name comes from an allowlist (`office_temp_c`, `office_humidity_pct`). Offline or stale rows are dropped. The read is a read-only transaction with a 10 s statement timeout.
- A young series no longer fails selection. Selection refuses before it runs the model zoo, and `/health` shows `not_ready` with `ready_at`. No `selection_failed` alert fires and no traceback is logged. Selection is re-checked at `ready_at`.
- The hour when the sensor was placed is excluded from training (the sensor was warm, not the room; see the gate below).
- An additive `(node, ts)` index on `office_env_sample` (CREATE INDEX CONCURRENTLY, auto-apply clean).
- The office series is NOT a cabinet covariate yet. A YAML comment says it joins once ~14 days of data exist (~2026-11-12).

## Outcome moved

Orion gets a forecast of the cabinet's intake air, the boundary condition of the cabinet heat balance that only outdoor weather stood in for until now. Expected first selection: ~2026-11-02 03:10 UTC (4.1 days of history: 2 days of training + 24 h horizon + 10 origins 3 h apart). Until then `/health` says when, instead of reporting a failure every 6 h.

## Current architecture

One target (`athena.cabinet_temp_c`), four source kinds (biometrics, cooling, cluster, weather). The engine already looped over `registry.targets()` with a per-series try/except. A series with less than ~4 days of history raised `ValueError: no baseline produced a residual` inside `select_champion`, which surfaced as `selection_failed` with a logged traceback every select interval.

## Architecture touched

- orion-prediction: registry source kind, store read, engine readiness gate, `/health` field.
- Series registry `config/prediction/series.v1.yaml`: one new target.
- orion-sql-db: one additive index migration.
- No bus, schema, or env changes.

## Files changed

- `config/prediction/series.v1.yaml`: `office.temp_c` target, placement exclude, covariate-later comment.
- `services/orion-prediction/app/series.py`: `office_env` source kind, `OFFICE_ENV_KEYS` allowlist, node pattern.
- `services/orion-prediction/app/store.py`: `office_env` read (filters, read-only + timeout via `_read_only`).
- `services/orion-prediction/app/backtest.py`: `min_backtest_steps` (history a backtest needs).
- `services/orion-prediction/app/engine.py`: `NotReady`, `check_ready` (history length, before the zoo runs), `check_paths` (residual paths after the backtest), not-ready back-off in `selection_due` and `tick_series`, `SeriesState.not_ready`.
- `services/orion-prediction/app/main.py`: `/health` series entry gains `not_ready`.
- `services/orion-prediction/tests/test_engine.py`: young target beside a mature one; ready-time lifecycle; `*` change events no-op.
- `services/orion-prediction/tests/test_series.py`: repo registry entry; bad office sources refused.
- `services/orion-prediction/tests/test_inputs.py`: office SQL shape (filters, read-only, timeout); store-side allowlist.
- `services/orion-sql-db/manual_migration_prediction_office_env_idx_v1.sql`: `(node, ts)` index.
- `services/orion-prediction/README.md`, `docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md`: targets table, young-series behaviour, metric gate.

## Schema / bus / API changes

- Added: `/health` -> `series.<id>.not_ready` (null, or `{reason, have_buckets, need_buckets, first_reading, ready_at}`).
- Added: forecasts for `office.temp_c` on `orion:prediction:forecast` (from ~11-02). The only consumer, orion-hardware-watch, filters on `series_id` (`services/orion-hardware-watch/app/watcher.py:220`), so it ignores them.
- Removed / renamed: none.
- Behavior changed: a series too young to backtest reports `not_ready`, not `selection_failed`.
- Compatibility notes: none; stored cabinet selections still restore (no gate-version change).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Metric quality gate (CLAUDE.md 0A)

1. **Provenance.** `services/orion-office-env/app/sample.py:18 build_sample` (via `app/main.py:54 make_sample`) turns the pan-1 Nano ESP32 SHT40 serial frame into `office.env.sample.v1`. A stale frame carries no measurements and `sensor.online = not stale`. `app/main.py:77 publish_sample` publishes it on `orion:office:env:sample`. orion-sql-writer `app/worker.py:1683 _normalize_office_env_sample_payload` flattens it into `office_env_sample`.
2. **Independence from `athena.cabinet_temp_c`.** Different sensor, board, host, producer and table. It is not causally independent, and that is the point: the room is the cabinet's intake air. Live, first 80 min after placement: cabinet minus room = 1.7..2.2 C, corr 0.94 over 8 ten-minute buckets. That is too few buckets to mean more than "they move together". As a target this does not matter. When it becomes a cabinet covariate, it is the boundary term, not a copy of the target.
3. **Theory anchor.** The cabinet heat balance: dT_cab/dt ∝ P_in − UA·(T_cab − T_room) − P_AC. T_room is the term the cabinet model was missing.
4. **Live sanity** (read-only, worktree code, 2026-10-29 01:25 UTC). 299 usable raw rows; 1 offline/stale row dropped. Raw 24.78..28.47 C. After the placement exclude: 15 five-minute buckets, 24.87..26.01 C, SD 0.39 C, 15 distinct values, no gaps, median |5-min step| 0.10 C, max 0.72 C. Not flat, not saturated. A calm room is a real rest state (the value is a temperature, not a deviation score). Readiness from the same run: 15 of 1188 buckets, `ready_at` 2026-11-02T03:10Z.
   - Placement exclude, 2026-10-28 22:16 -> 2026-10-29 00:10: 22:16-22:35 at a first spot (RH ~30%), no data 22:35-23:10, then 28.2 C falling to 25.3 C while RH rose 21.7 -> 24.5%. Vapour pressure stayed about the same (~0.83 vs ~0.79 kPa), so the sensor was cooling, not the room.
5. **Existing mechanism.** Hub's climate overlay (`services/orion-hub/scripts/cabinet_climate_routes.py`, ef270222d) only charts it. Nothing forecast it.
6. **Reversibility.** One registry entry, one read-only source kind, one additive index. Deleting the entry stops it. Its stored rows are keyed by `series_id` and harm nothing.

## Tests run

```text
(service image orion-prediction-prediction + pytest/httpx; host venv lacks prophet/statsforecast)
cd services/orion-prediction && PYTHONPATH=<worktree> python -m pytest -p no:cacheprovider tests -q
  176 passed
Mutation check: check_ready() call removed -> the two young-series tests fail (2 failed, 1 passed);
  restored -> pass.
python scripts/check_migration_required_by.py   -> OK (1 new migration file)
python scripts/check_env_key_single_source.py   -> OK
python scripts/apply_post_pull_migrations.py --dry-run <new file>
  -> NOT APPLIED; would apply (no transaction wrapper: CREATE INDEX CONCURRENTLY, lock_timeout=10s)
git diff --check -> clean
```

## Evals run

```text
evals/run_causal_calibration_eval.py -> PASS (switchback coverage95 0.97, null FPR 0.05; ITS coverage95 0.95)
evals/run_cabinet_replay_eval.py     -> RESULT: PASS (40 windows, committed real cabinet fixture; cabinet
  assignment naive @5/15/60/1440, window_average_1h @360; unchanged by this PR -- backtest code untouched)
```

## Docker/build/smoke checks

```text
Live read-only smoke (worktree code in the service image, app-net, real POSTGRES_URI):
  raw rows=299 first=2026-10-28 22:16:06 last=2026-10-29 01:24:36 min=24.78 max=28.47
  5-min buckets after exclude=15 observed=15 gaps=0; 24.87..26.01 C, SD 0.387, 15 distinct
  not_ready: 15 of 1188 buckets (backtest needs 4.1 days), ready_at 2026-11-02T03:10:00+00:00
EXPLAIN ANALYZE of the office read today: 2.0 ms (304 rows, seq scan; table is tiny).
Not deployed (deploy runs from primary main after merge).
```

## Review findings fixed

Review ran in a subagent on `feat/prediction-office-temp` vs `origin/main`.

- Finding (must-fix): `ready_at` ignored the 60 s settle delay. The first tick at `ready_at` refused again and fell back to a 6 h wait. The test jumped to `ready_at + 90 s` and so passed for the wrong reason.
  - Fix: `ready_at` now includes the settle. If a series is still refused at or after `ready_at`, it re-checks one grain later, not one interval later.
  - Evidence: the test now ticks at `ready_at - 30 s` (still waiting, wakes at `ready_at`) and at `ready_at` (selects). Mutation (settle removed): 2 tests fail.
- Finding (should-fix): the history gate counts gaps as history and has zero margin. A gappy young series could pass it, then end in `selection_failed`, or serve a selection whose every issue is refused as degenerate.
  - Fix: `check_paths` runs after the backtest. Every served horizon needs 10 residual paths in some baseline, and then in both blends. Too few with nothing served gives `not_ready` (`ready_at` null, re-check every 6 h). Too few with a selection already served gives a failed reselect, which keeps the old selection.
  - Evidence: new test `test_a_young_target_old_enough_but_gappy_is_not_ready_not_failed_or_degenerate`. Mutation (both checks removed): the test fails because a selection that can never issue gets served.
- Finding (should-fix): not-ready did not back off in `selection_due`. A stale served selection plus a young series would reselect, reload ~27 days of history and refuse again every 30 s tick.
  - Fix: `not_ready` counts as backing off, the same as `last_select_error`.
  - Evidence: new test `test_not_ready_backs_off_even_when_a_stale_selection_would_force_a_reselect`: 4 ticks, 1 check, old selection kept. Mutation: the test fails.
- Finding (nit): `_read_only` duplicated the event-table read-only block.
  - Fix: `event_times` now uses `_read_only`.
  - Evidence: event-pair tests pass (full suite 176 passed).

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

Run from the clean primary main checkout after merge. The post-pull rebuild applies the index migration first. By hand: `docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_prediction_office_env_idx_v1.sql`.

## Risks / concerns

- Severity: low
  - Concern: the `(node, ts)` index adds nothing measurable today (one node, ~3k rows/day; the `ts` index and a seq scan both serve it in ~2 ms). It future-proofs a second room sensor, and it makes the deploy gate wait for the migration.
  - Mitigation: additive, CONCURRENTLY, auto-applied post-pull; rollback is one `DROP INDEX CONCURRENTLY`.
- Severity: low
  - Concern: after `ready_at`, gaps can still leave fewer than 10 usable backtest windows. The series then shows `selection_failed` or `degenerate_forecast` (existing paths) until enough clean history exists.
  - Mitigation: those paths are already alerting and backed off; the cabinet target is isolated per series.
- Severity: low
  - Concern: UNVERIFIED live: the first real `office.temp_c` selection and forecast (~2026-11-02) have not happened.
  - Mitigation: check `/health` `series.office.temp_c` after deploy (expect `not_ready`), then after 11-02 03:10 UTC.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2956

🤖 Generated with [Claude Code](https://claude.com/claude-code)
