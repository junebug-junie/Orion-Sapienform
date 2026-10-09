## Summary

- Save the pressure calculated at every dream check, including checks where no dream runs.
- Save the GPU pool's existing five-second state broadcasts as small backlog/queue history rows.
- Keep failed reads, absent data and stale snapshots distinguishable from calm.
- Extend the existing read-only reports to consume these histories. No dream, GPU, focus or arousal decision rule changes.

## Outcome moved

The missing evidence found in #2561 can now accumulate. A future report can see pressure discharge and recovery between dreams, and use actual GPU snapshots instead of reconstructing backlog from sparse lease events.

**Production capture is UNVERIFIED until deployment.** This branch has not migrated or restarted production. Local proof includes real PostgreSQL round trips, unchanged-decision fixtures, and three genuine live GPU broadcasts passed through the new writer into an isolated test database.

## Current architecture

- `orion-dream` computes pressure in `app/cycle.py:read_pressure`, but previously saved it only inside a completed/failed/empty cycle. Its entry points are the periodic loop and manual cycle endpoint; the HTTP pressure endpoint remains a read, not a scheduler observation.
- `orion-gpu-pool/app/runtime.py:snapshot` counts leases whose status is `backlogged` or `queued`, and `publish_state` broadcasts `GpuPoolStateV1`. The SQL writer previously persisted lease events but not these snapshots.
- Existing SQL writer routing, typed validation, insert-only writes, write-health reporting, and bounded periodic retention are reused.
- Both services have README, `.env_example`, compose, requirements, tests and evals; settings are in `app/settings.py`, not service root. No dependency is added.

## Architecture touched

1. Contract: SQL-only `DreamPressureObservationV1` registered alongside the existing dream schemas, plus two additive SQL tables. The existing GPU bus payload is unchanged; SQL writer becomes a declared consumer.
2. Producer: dream records the exact `SleepPressureV1` already computed, prior cycle clocks, settings, formula family and source-read errors before evaluating the existing gates. A failed source/clock query invalidates its observation without changing the old empty/None/cache scheduling fallback. Recording failure logs and continues.
3. Consumer: SQL writer validates the existing GPU payload and stores only source time, host, mode, configuration digest, queue and backlog. Explicit source time/host/depth fields are required; model defaults cannot manufacture fresh zero readings. Stable host/time/config identity makes redelivery idempotent.
4. Retention: dream keeps 30 days, deleting at most 1,000 expired rows per check in a separate transaction. GPU history has a 30-day default using the existing bounded retention loop and health surface. The loop's budget rises from 66 to 70 seconds to preserve its tested five-second floor across fourteen tables.
5. Offline consumers: dream `--with-checks` exports observations alongside cycles; arousal `--gpu-host circe` exports one source's actual snapshots. Mixed hosts are rejected. Legacy exports still work.

### Metric quality record

- **Provenance:** pressure is produced by `app/replay.py:compute_pressure` through `cycle.py:read_pressure`; the observer stores that same object before the gates. Backlog is the count of `live_leases()` rows with status `backlogged` in GPU `runtime.py:snapshot`. Queue is counted separately. The writer stores these emitted fields verbatim.
- **Independence:** these histories duplicate existing measurements for inspection. They are explicitly redundant observations, not new independent inputs. No consumer in a cognition model, drive or action loop is added.
- **Anchor:** queue/backlog depth is an exact queue-state count, not a fatigue score. Dream pressure retains its existing weighted-new-material definition; this patch makes no new physiological or need claim. Arousal remains an offline hypothesis.
- **Live sanity:** at 2026-10-09 05:53:27.070677, 05:53:32.127229 and 05:53:37.182364 UTC, the real pool identified itself as `circe`; backlog was `{}`, while queue was `agent=5`, `diffusion=1`, `memory_distill=1`. Thus an empty backlog is a real emitted state and is not equivalent to no queued work. These three payloads produced three SQL rows in the isolated database. They do not establish a varied long-term curve or sustained-strain frequency. The live read-only dream endpoint at 05:53:37.222025 returned novelty pressure **13.262**, threshold **3**, and new counts `metacog=17`, `crystallization=1`. That single HTTP reading does not establish discharge/recovery and was not labelled a scheduler check.
- **Existing mechanisms:** reuse the dream loader/pressure object, GPU broadcast/schema, writer route/validator/idempotency, retention loop, and #2561 reports. No parallel sensor, timer, regulator or bus channel.
- **Reversibility:** stop these recorders by reverting this branch's service/config changes, retaining the tables for inspection. Remove the added GPU subscription/route entries from local env when rolling the writer back. No destructive cleanup is needed. No training, manifest or decision default consumes these histories.

The metric-definition lock records exactly one routing change: adding SQL writer to `orion:gpu_pool:state`. It does not introduce a new metric definition.

## Files changed

- `orion/schemas/dream_cycle.py`, `registry.py`: observation contract and registration.
- `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`: declare and acknowledge the existing state's additional consumer.
- Dream `app/cycle.py`, `cycle_store.py`, `main.py`: observe checks and read validity, append to bounded history.
- SQL writer model, worker, settings, retention modules, `.env_example`, compose: persist minimal state, wire subscription/config/retention.
- `services/orion-sql-db/manual_migration_regulation_history.sql`: additive, repeatable table/index creation.
- The two measurement scripts and their tests: read the new histories without changing Orion.
- Dream/writer tests, dream eval, service READMEs and `.github/workflows/regulation-history.yml`: verification and operating instructions.

## Schema / bus / API changes

- Added: `DreamPressureObservationV1` (SQL-only); `dream_pressure_observation`; `gpu_pool_state_history`.
- Removed/renamed: none.
- Bus: existing `gpu_pool.state.v1` shape unchanged; SQL writer added to `orion:gpu_pool:state` consumers.
- Behavior changed: observation persistence only. No new HTTP endpoint and no new action/regulation reader.
- Compatibility: old cycle-only and lease-event-only reports remain supported. Apply the migration before restarting dream. Missing migration logs a recording failure and does not block sleeping.

## Env/config changes

- Added key: `GPU_POOL_STATE_HISTORY_RETENTION_DAYS=30` in settings, template and compose.
- Changed default: `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC`, 66 → 70; compose now passes it explicitly.
- Added structured entries: state subscription and route in `SQL_WRITER_SUBSCRIBE_CHANNELS` / `SQL_WRITER_ROUTE_MAP_JSON`.
- Removed/renamed keys: none.
- `.env_example` updated: SQL writer only.
- Local `.env`: synced from this worktree using `python3 scripts/sync_local_env_from_example.py` and the writer's `--all-keys` pass. The known old 66-second local budget was raised to the new 70-second floor; structured entries are present. Env parity passes. `.env` is ignored and uncommitted.
- Protected sync keys, including `ORION_BUS_URL` and `PUBLISH_CORTEX_EXEC_GRAMMAR`, were preserved. No task-related skipped key needs operator action. Live bus reads used the configured Tailscale address.

## Tests run

```text
Dream cycle + pressure history: 44 passed, including isolated PostgreSQL.
Writer state/event + existing retention + route-map checks: 82 passed, including isolated PostgreSQL.
Offline report checks: 21 passed.
Env parity: PASS.
git diff --check: clean.
```

The disposable `regulation_history_test` database is isolated from `conjourney`. Tests explicitly reject any other database name. Production tables were only read; neither the migration nor history writes ran against production.

## Evals run

```text
services/orion-dream/evals/test_dream_cycle_eval.py: 5 passed.
New trace eval: due pressure -> completed sleep -> recorded zero while refractory -> recorded one
after new material, with no extra sleep.
Live GPU consumer smoke: 3 captured broadcasts -> worker._write -> 3 isolated PostgreSQL rows.
```

## Docker/build/smoke checks

Both affected images were built through `scripts/safe_docker_build.sh`. Network-disabled image import checks verify dream's observer is wired and SQL writer's state route/subscription exists. These are build/integration proofs, not production capture claims.

## Review findings fixed

- Finding: failed previous-cycle clock reads looked like valid `None` in observation history.
  - Fix: errors travel through the real cache wrappers into the observation while preserving the original scheduling fallback.
  - Evidence: `test_clock_query_failures_survive_real_cache_wrappers_and_reach_history`, plus independent review.
- Finding: fourteen retention tables exceeded the existing per-table budget floor.
  - Fix: raise cycle budget to 70 seconds and sync settings/template/compose/local env; expose GPU retention in health with an explicit settings field.
  - Evidence: existing retention tests and new history tests pass.

The requesting-code-review skill ran in an independent subagent. Re-review approved the code with no material findings remaining.

## Restart required

Deployment commands, **not executed against production**:

```bash
cd /mnt/scripts/Orion-Sapienform-regulation-history
docker exec -i orion-athena-sql-db psql -X -v ON_ERROR_STOP=1 -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_regulation_history.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
```

No GPU pool or attention restart is needed. After deployment, verify source timestamps move, not merely container health:

```sql
BEGIN READ ONLY;
SELECT observed_at, observation_json->'reading'->>'pressure' AS pressure,
       observation_json->'source_errors' AS source_errors
FROM dream_pressure_observation ORDER BY observed_at DESC LIMIT 3;
SELECT generated_at, host, backlog_depth, queue_depth
FROM gpu_pool_state_history ORDER BY generated_at DESC LIMIT 3;
ROLLBACK;
```

Expect one pressure observation per completed scheduler check (normally about ten minutes plus work time) and GPU snapshots about five seconds apart. Repeat the query to prove cursor/time movement. Then run the two offline reports using `--with-checks` / `--gpu-host circe`.

## Risks / concerns

- Medium: production capture remains **UNVERIFIED** until the migration and two restarts. No behavior rollout is justified yet.
- Low: recording adds bounded database work. Dream's separate small pool and short timeouts keep failures isolated; writes and cleanup can still add several seconds to a check under database failure.
- Low: Redis pub/sub cannot replay an outage. Source-time freshness and cadence gaps remain visible; reports never interpolate missing observations.
- Low: historical source caps and formula semantics remain those of the existing pressure producer. Recording creates evidence, not a promise that the pressure is a valid rest drive.

## PR link

[PR #2563](https://github.com/junebug-junie/Orion-Sapienform/pull/2563)
