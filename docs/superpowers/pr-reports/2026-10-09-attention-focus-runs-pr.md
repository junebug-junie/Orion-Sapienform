## Summary

- Record one `field_dominance_run` when the existing goal-provenance focus target changes or disappears, including one-tick runs.
- Persist the open run across restarts; deduplicate frames and mark an already-running first observation as left-censored.
- Fully remove streak-tick telemetry code/config and prepare the old-table retirement migration.
- Keep selection, competition hysteresis, debounce and goal payloads unchanged. Recorder failures cannot suppress goals.

## Outcome moved

There is now a direct, inspectable record of each observed focus interval. A seven-day export taken 2026-10-09 replayed **284,285 ticks into 4,675 closed runs**, exactly matching an independent grouping of the selected target sequence. This is the existing **internal-signal goal-provenance winner**, not the attention frame's overall winner or proof of busy/idle/arousal. No decision consumer is added.

## Current architecture

- `orion-attention-runtime/app/main.py` starts the polling worker. Settings live in `app/settings.py` (no root `settings.py`); `.env_example`, compose, requirements, Dockerfile and service tests exist.
- Each new field tick builds/persists an attention frame, advances the persisted `DominanceStreak` debounce and optionally emits `FieldGoalProvenanceV1` on `orion:memory:goals:proposed`.
- A separate per-tick debug event previously crossed Redis to SQL-writer's `DominanceStreakTickSQL` table. Its only analysis client queried that table.
- Attention-runtime had no eval directory; this patch adds the small replay harness. SQL-writer already has tests/evals.

## Architecture touched

Attention-runtime records directly through its existing Postgres store. On success, frame, completed row, run checkpoint and debounce commit together. Nested savepoints isolate recorder and debounce persistence failures, preserving the existing decision path. Saved frames reveal a skipped recorder tick after restart; recovery logs the gap and abandons an uncertain open run rather than fabricating continuity.

## Files changed

- `orion/schemas/field_dominance_run.py`, `registry.py`: SQL-only row contract.
- `services/orion-attention-runtime/app/{dominance_runs,store,worker}.py`: segment selected winners, persist checkpoint/history, remove telemetry publishing.
- Attention README/settings/env/compose/tests/evals and `.github/workflows/orion-attention-runtime-tests.yml`: contract, retirement, replay and real Postgres CI.
- `services/orion-sql-db/manual_migration_field_dominance_run_v1.sql`: history table plus checkpoint column.
- `services/orion-sql-db/manual_migration_retire_streak_tick_v1.sql`: separately approved old-table removal, without CASCADE.
- `orion/bus/channels.yaml`, `orion/schemas/field_goal.py`, SQL-writer app/models/settings/env/README: retire the debug rail and boot retention/DDL.
- Deleted the old live analysis query and its tests; the new eval reads an offline export.
- `config/metrics/metric_definitions.lock.json`: regenerated to record the removed bus instrument.
- Adjacent dev-economics/retention comments no longer refer to deleted code.

## Schema / bus / API changes

- Added: registered `FieldDominanceRunV1`, `field_dominance_run`, nullable `substrate_goal_provenance_streak.run_state` checkpoint.
- Removed: `DominanceStreakTickV1`, `debug.attention.streak_tick.v1`, `orion:debug:attention:streak_tick`, writer model/routes/subscription/boot DDL/retention; table removal is a prepared manual migration, **not applied to production**.
- Renamed: none.
- Behavior changed: observational history only; no ranking, threshold, goal payload or action changes.
- Compatibility: apply additive migration before attention deployment, update both services, verify completed rows, then drop the old table. Keeping the existing decision debounce is necessary to preserve behavior; it is distinct from the deleted per-tick telemetry.
- `ended_at` is the transition frame timestamp. Last-source-frame id belongs to the old run. No-winner ticks close runs without opening fake targets. The open tail is not a completed row.
- A left-censored row counts only observed ticks after installation/recovery. Wall-clock spans can include downtime; absence of rows is not evidence of rest.

### Metric quality gate (before any decision consumer)

1. **Provenance:** `worker._maybe_build_goal()` reuses the unchanged `top_node_substrate_target()` selection and `update_dominance_streak()` debounce. `app/dominance_runs.py:advance_run()` counts each distinct observed field tick. Timestamps and frame ids come from the actual built `FieldAttentionFrameV1`; SQL writes are in `store._save_dominance_tick()`.
2. **Independence:** this is a run-length encoding of the existing winner sequence, causally derived from the same ranking/competition inputs. It is redundant with that sequence, **not an independent need, activity or arousal signal**. No combined model consumes it.
3. **Anchor:** exact event segmentation by target identity, with counting measure over observed ticks and elapsed wall-clock time between observed transition boundaries. No psychological detector or claim is made. A named cognitive theory is not required to count events; interpreting these intervals as rest/strain would require a separate gate.
4. **Live sanity:** seven-day source export has five non-null targets, two no-winner ticks, observed closed-run lengths 1–1,513 ticks (median 22). The source is not flat. No-winner closes the run; no floor, decay or threshold invents a winner. This checks recording, **not a calm-state interpretation**. Actual new production rows remain **UNVERIFIED** until rollout.
5. **Existing mechanism:** existing `DominanceStreak`, singleton persistence, frame writer, legacy telemetry schema/model/channel and offline query were inspected. The same winner is reused; legacy instrumentation is retired, not hidden from one reader.
6. **Reversibility:** no decision consumer or training default. Revert code/redeploy and keep additive history untouched. Dropping old history is irreversible without an export, so retirement is a separate operator-approved migration.

## Env/config changes

- Added keys: none. Renamed keys: none.
- Removed: `ORION_GOAL_PROVENANCE_STREAK_TICK_TELEMETRY_ENABLED`, `CHANNEL_GOAL_PROVENANCE_STREAK_TICK`, `GOAL_PROVENANCE_STREAK_TICKS_RETENTION_DAYS`, old entries in SQL-writer subscribe/route JSON.
- Both `.env_example` files updated; ran `python3 scripts/sync_local_env_from_example.py orion-attention-runtime orion-sql-writer` and removed obsolete keys/JSON entries from ignored primary-checkout `.env` files (the sync helper does not remove them). Copied the synced files into the worktree for builds.
- Worktree-template parity passed. `ORION_BUS_URL` is inherited from root `.env`; established Tailscale address verified and preserved. No skipped changed keys require manual action. `.env` files verified ignored.
- Existing tooling limitation: `check_env_template_parity.py` compares the primary checkout's **old** template for existing services, so it rejects intentional retirement before main advances. Independently called its `check_service()` on both worktree service directories and ran the hostname gate; both passed. Used documented `ORION_ALLOW_ENV_DRIFT=1` for the SQL-writer **build only**, not a deploy. Follow-up: make this gate read branch templates like the sync helper already does.

## Tests run

```text
FOCUS_RUN_TEST_POSTGRES_URI=<disposable test DB> python -m pytest -q \
  services/orion-attention-runtime/tests tests/test_attention_runtime_worker.py \
  tests/test_attention_runtime_store.py tests/test_attention_field_goal_provenance.py
  -> 97 passed (includes real PostgreSQL recording, failure recovery and retirement)
PYTHONPATH=.:services/orion-sql-writer python -m pytest -q \
  services/orion-sql-writer/tests/test_route_map_completeness.py \
  services/orion-sql-writer/tests/test_consumer_resilience.py \
  services/orion-sql-writer/tests/test_dev_economics_ledger_sql_shape.py
  -> 15 passed
python scripts/check_metric_lineage.py --gate -> PASS
python scripts/check_definition_drift.py --gate -> PASS (rechecked after commit)
python scripts/check_inner_state_registry.py -> PASS
python scripts/check_service_env_compose_parity.py <each touched service> -> PASS / env_file
Worktree env parity, hostname checks, retirement scan, git diff --check -> PASS
```

## Evals run

```text
python -m pytest -q services/orion-attention-runtime/evals -> 1 passed
python -m pytest -q services/orion-sql-writer/evals/test_storage_write_replay_eval.py -> 3 passed
python services/orion-attention-runtime/evals/replay_focus_runs.py /tmp/orion-focus-history.jsonl
  -> exact match: 284,285 ticks, 4,675 closed runs; simulated restarts every 997 ticks
```

Export and report are host-local `/tmp/orion-focus-history.jsonl` and `/tmp/orion-focus-replay-result.json`; no production writes or backfill.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-attention-runtime build -> PASS
ORION_ALLOW_ENV_DRIFT=1 scripts/safe_docker_build.sh orion-sql-writer build -> PASS (reason above)
Built attention image -> real worker/store -> disposable PostgreSQL: two closed rows and unchanged goal emission
Built SQL-writer image: imports succeed; retired route/model/subscription absent
```

Production path: **UNVERIFIED**. Production services were not restarted and neither migration was applied there.

## Review findings fixed

- Finding: a recorder-specific SQL error could prevent attention frames and goals.
  - Fix: isolate recording with a savepoint; preserve decision persistence behavior; detect skipped recording ticks from saved frames across restarts.
  - Evidence: PostgreSQL tests inject a recording constraint failure and remove access to the new checkpoint column; frames/debounce/goals survive. Recovery logs a gap and marks the resumed partial run. Independent review repeated: **No findings**.

## Restart required

After operator approval, from this reviewed worktree (update main's template after merge or use the disclosed branch-parity check before the SQL-writer override):

```bash
docker exec -i orion-athena-sql-db psql -v ON_ERROR_STOP=1 -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_field_dominance_run_v1.sql
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
curl -fsS http://localhost:8117/health
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c \
  'SELECT target_id, started_at, ended_at, tick_count FROM field_dominance_run ORDER BY ended_at DESC LIMIT 10;'
# Only after a real transition lands and the retired writer is stopped:
docker exec -i orion-athena-sql-db psql -v ON_ERROR_STOP=1 -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_retire_streak_tick_v1.sql
```

Table-drop approval is required by `AGENTS.md` §13; old historical data must be exported first if it is to be retained. The seven-day read-only replay export exists already; it is not a full-table backup.

## Risks / concerns

- Severity: medium. Production rollout/table retirement are pending approval; the old deployed producer/table still run until then. Do not call runtime retirement verified yet.
- Severity: low. Recorder failures lose uncertain intervals with explicit logs, preserving attention decisions. Gaps and censored runs must be excluded from complete-duration claims.
- Severity: low. These selected internal-signal runs are not a record of all overall attention winners, human activity or continuous uptime. Later habituation/rest/arousal work must pass its own measurement gate.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2555
