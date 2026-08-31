# Declare the Postgres connection ceiling, and watch it

## Summary

- Postgres was running on the **postgres:15 default `max_connections=100`**, never declared anywhere in the repo. 25 distinct clients had grown into it.
- Live reading on 2026-08-31: **94 client backends against a service-visible ceiling of 97**. `psql` was refused three times with `FATAL: sorry, too many clients already` during an unrelated investigation.
- Declared the ceiling as `POSTGRES_MAX_CONNECTIONS` (default 300) in the sql-db compose and `.env_example`.
- Added `scripts/check_postgres_connection_headroom.py`, a gate that measures against the ceiling services actually hit, and treats *its own refusal by saturation* as the alarm rather than as a crash.
- Deliberately did **not** add `idle_session_timeout`. Rationale below.

## Outcome moved

A silent capacity ceiling became a declared one with a check behind it. Before: nothing measured connection headroom, and exhaustion was found by accident while investigating something else. After: `check_postgres_connection_headroom.py --gate` fails at <15% free, and reports rather than dies when the database is already full.

Note the ceiling change itself is **not live** — `max_connections` is postmaster-only and needs a full DB restart (see Restart required).

## Current architecture

`services/orion-sql-db/docker-compose.yml` runs `postgres:15` with a `command:` list that already declares `shared_preload_libraries`, `autovacuum_work_mem`, `maintenance_work_mem`, `max_parallel_maintenance_workers` and `vacuum_cost_delay`. `max_connections` was absent from that list, from `.env_example`, and from every other config surface in the repo — so it silently took the image default of 100.

No script, gate, or health check read connection counts.

## Architecture touched

- `services/orion-sql-db` compose `command:` list and `.env_example`.
- `scripts/` gains one read-only diagnostic. No service code, no bus channel, no schema.

## Files changed

- `services/orion-sql-db/docker-compose.yml`: declare `-c max_connections=${POSTGRES_MAX_CONNECTIONS:-300}`, with the measurement and the sizing rationale in a comment matching the file's existing style.
- `services/orion-sql-db/.env_example`: add `POSTGRES_MAX_CONNECTIONS=300` and note it needs a full restart.
- `scripts/check_postgres_connection_headroom.py`: new. Reports `used/service_ceiling`, an idle-vs-stale split, and top clients.
- `tests/test_check_postgres_connection_headroom.py`: new, 22 tests.

## Schema / bus / API changes

None. No channel, schema, or payload touched.

## Env/config changes

- Added keys: `POSTGRES_MAX_CONNECTIONS` (`services/orion-sql-db/.env_example`, default `300`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: **`scripts/sync_local_env_from_example.py` could not do it.** It reads `.env_example` from the *primary* checkout, so a key added in a worktree is invisible to it. The key was written into the live `services/orion-sql-db/.env` by hand and verified still gitignored (`git check-ignore` → ignored).
- skipped keys requiring operator action: none

## The measurement

Two things made the raw number misleading, and both are now handled:

**1. Services never get all of `max_connections`.** `superuser_reserved_connections` (3) is held back, so the ceiling a normal service hits is 97, not 100. At the time of the incident the service-visible pool was entirely gone while `max_connections` still read "100".

**2. `pg_stat_activity` counts things that hold no connection slot.** The first version of this script reported `102/97 used` — impossible. The live server had 91 client backends plus a checkpointer, walwriter, background writer, autovacuum launcher and logical replication launcher. Those five are budgeted by `max_worker_processes` and friends, not `max_connections`. All three `pg_stat_activity` reads now filter `backend_type = 'client backend'`. This was caught by looking at real output, not by review.

## Why this is not a leak, and why there is no idle_session_timeout

Of 93 idle connections, **only 8 had been idle longer than two hours**. The rest were live pool connections cycling normally — last statement `COMMIT` or `ROLLBACK`, `state_change` seconds old. ~25 services each holding a SQLAlchemy pool (default `pool_size=5` + `max_overflow=10`) can legitimately want ~375; `orion-sql-writer` alone is configured to burst to 38.

Reaping idle sessions was considered and rejected: **18 of the 80** `create_engine`/`create_async_engine` calls in `orion/` and `services/` do not set `pool_pre_ping` (AST-counted, tests excluded), including live paths in `orion/substrate/mutation_queue.py` and `orion/substrate/policy_profiles.py`. Killing idle sessions would convert a capacity problem into scattered pool-checkout errors in exactly those pools — trading a visible ceiling for an intermittent one.

> **Correction.** Commit `2c7cfd916` and an earlier draft of this report said `pool_pre_ping` was set at "only 38" sites. That came from a line-based grep, which misses multi-line calls and made coverage look far worse than it is: the real split is 62 with, 18 without (78% covered). The decision does not change — 18 unprotected pools on live paths is reason enough not to reap idle sessions — but the number backing it was wrong.

## Tests run

```text
pytest tests/test_check_postgres_connection_headroom.py -q
22 passed in 0.15s
```

Mutation-tested against the real file, each mutation asserted to actually apply before running (a replacement that silently fails to match reads as a false green):

```text
RED | drop reserved subtraction              | 6 failed, 14 passed
RED | saturation matcher always true         | 3 failed, 17 passed
RED | drop message fallback (sqlstate only)  | 2 failed, 18 passed
RED | re-raise instead of reporting saturation| 1 failed, 19 passed
RED | drop the free-slot clamp               | 1 failed, 19 passed
RED | headroom counts all backends           | 2 failed, 20 passed
RED | idle split counts all backends         | 1 failed, 21 passed
RED | top_clients counts all backends        | 1 failed, 21 passed
```

All ten static gates named in `.github/workflows/orion-static-gates.yml` (list derived from CI, not from memory):

```text
OK check_compose_no_relative_mounts.py    (83 compose files, 0 relative host mounts)
OK check_control_surface_store_parity.py
OK check_daily_schedule_collisions.py
OK check_definition_drift.py --gate
OK check_inner_state_registry.py
OK check_journal_dispatch_registry.py
OK check_metric_lineage.py --gate
OK check_scripts_dir_no_stdlib_shadow.py
OK check_service_hostname_refs.py
OK check_system_health_producers.py
```

## Evals run

No eval harness exists for `services/orion-sql-db` (the directory holds compose, env, and migration SQL only — no `tests/` or `evals/`). This change adds a repo-root script and its tests rather than service code, so no eval harness was created. Not claiming eval coverage.

## Docker/build/smoke checks

Compose renders the new setting from the live env files:

```text
docker compose --env-file .env --env-file services/orion-sql-db/.env \
  -f services/orion-sql-db/docker-compose.yml config
  25:      - shared_preload_libraries=pg_stat_statements
  27:      - max_connections=300
  42:    mem_limit: "17179869184"
```

Live run of the gate against the running database, before any restart:

```text
postgres connections: 94/97 used (3 free, 3%) [max_connections=100, reserved=3]
  idle: 93 total, 8 idle >2h (a leak looks like a large second number)
  172.18.0.1           22
  172.18.0.69          14   (orion-athena-sql-writer)
  172.18.0.2            7   (orion-athena-thought)
  172.18.0.18           7   (orion-athena-substrate-runtime)
```

Exit code 1 under `--gate`, as intended.

## Restart required

`max_connections` is postmaster-only. This does **not** take effect until orion-sql-db is fully restarted, and that restarts the database every Orion service depends on.

```bash
cd /mnt/scripts/Orion-Sapienform
docker compose --env-file .env --env-file services/orion-sql-db/.env \
  -f services/orion-sql-db/docker-compose.yml up -d --force-recreate orion-sql-db
```

Then confirm the ceiling actually moved:

```bash
POSTGRES_URI="postgresql://postgres:<pw>@localhost:55432/postgres" \
  python3 scripts/check_postgres_connection_headroom.py --verbose
# expect: ... [max_connections=300, reserved=3]
```

Before restarting, check no backup or long migration holds a lock — a prior incident had boot sit 35 minutes at "Waiting for application startup" behind a backup.

## Risks / concerns

- **Severity: medium.** Restarting Postgres takes every dependent service's connections down. Services with `pool_pre_ping` recover; the ones without it may need their own restart. Worth doing during a quiet window.
- **Severity: low.** 300 backends cost roughly 1–3GB of backend memory against the container's 16g `mem_limit`, with `shared_buffers` still at the 128MB default. Comfortable, but it is a real increase.
- **Severity: low.** Raising the ceiling relieves the symptom and does not reduce demand. If usage keeps climbing, 300 buys time rather than solving pool sizing. The gate is what turns the next approach into a warning instead of an outage.
- **Observed, not fixed:** `shared_buffers` is at the postgres default of 128MB on a 16GB container. That is almost certainly too small, but changing it is a separate performance decision needing its own evidence, not a rider on this patch.

## Status

DONE_WITH_CONCERNS — the code and the gate are complete and verified, but the ceiling is not live until Juniper restarts orion-sql-db.
