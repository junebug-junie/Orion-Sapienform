# Declare the Postgres connection ceiling, and watch it

## Summary

- Postgres was running on the **postgres:15 default `max_connections=100`**, never declared anywhere in the repo. The server log holds **217 `FATAL: sorry, too many clients already` over five days** (2026-08-26 → 08-31). This is an ongoing condition, not a one-off.
- Declared the ceiling as `POSTGRES_MAX_CONNECTIONS` (default 300) in the sql-db compose and `.env_example`.
- Added `scripts/check_postgres_connection_headroom.py` + `make postgres-headroom`: reports headroom, exits 1 on alarm and 2 when it cannot check, and treats *its own refusal by saturation* as the alarm rather than a crash.
- **Surfaced a hazard the patch does not fix:** every client backend connects as `postgres`, a superuser, so `superuser_reserved_connections` holds nothing back. When the database fills there is no emergency door for an operator. The check now warns about this explicitly.
- Deliberately did **not** add `idle_session_timeout`.

## Outcome moved

A silent, recurring capacity ceiling became a declared one with an instrument behind it. Before: nothing measured connection headroom; 217 refusals accumulated unnoticed and the condition was found by accident. After: `make postgres-headroom` reports and alarms, and names the missing emergency door.

The ceiling change is **not live** — `max_connections` is postmaster-only and needs a full DB restart.

## Current architecture

`services/orion-sql-db/docker-compose.yml` runs `postgres:15` with a `command:` list already declaring `shared_preload_libraries`, `autovacuum_work_mem`, `maintenance_work_mem`, `max_parallel_maintenance_workers`, `vacuum_cost_delay`. `max_connections` was absent from that list, from `.env_example`, and from every other config surface. No script or health check read connection counts.

## Architecture touched

`services/orion-sql-db` compose `command:` and `.env_example`; one read-only script in `scripts/` with a Makefile target. No service code, bus channel, or schema.

## Files changed

- `services/orion-sql-db/docker-compose.yml`: declare `-c max_connections=${POSTGRES_MAX_CONNECTIONS:-300}`, with the log evidence, the superuser-ceiling correction, the sizing reconciliation and the memory analysis in a comment.
- `services/orion-sql-db/.env_example`: add `POSTGRES_MAX_CONNECTIONS=300`.
- `scripts/check_postgres_connection_headroom.py`: new.
- `tests/test_check_postgres_connection_headroom.py`: new, 29 tests (2 live, skipped without a database).
- `Makefile`: `postgres-headroom` target.

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: `POSTGRES_MAX_CONNECTIONS` (`services/orion-sql-db/.env_example`, default `300`)
- Removed / renamed: none
- `.env_example` updated: yes
- local `.env` synced: **`scripts/sync_local_env_from_example.py` cannot do it** — it reads `.env_example` from the *primary* checkout, so a key added in a worktree is invisible to it. Written into the live `services/orion-sql-db/.env` by hand; `git check-ignore` confirms it is still ignored.
- skipped keys requiring operator action: none

## Which ceiling actually applies

The naive reading is `max_connections - superuser_reserved_connections` = 97. **That is wrong for this deployment**, and the log proves it rather than inferring it:

```text
sorry, too many clients already .................................. 217
remaining connection slots are reserved for non-... superuser ...... 0
```

The second message is what an *ordinary* role receives at the lower ceiling. Zero occurrences across 217 real refusals. Confirmed directly:

```text
SELECT a.usename, u.usesuper, count(*) ... WHERE backend_type='client backend'
postgres | t | 70
```

Every client backend is the `postgres` superuser. The wall is 100.

**The hazard this exposes:** `superuser_reserved_connections` exists so an operator can always get in during an incident. Services connecting *as* the superuser spend that reserve like any other slot — which is exactly why `psql` was refused three times while diagnosing the original problem. Raising the ceiling buys room; moving services to a non-superuser role is the real fix, and is not attempted here.

## Why this is not a leak, and why there is no idle_session_timeout

Of 93 idle connections, **only 8 had been idle longer than two hours**. The rest were live pool connections cycling normally. ~25 services each holding a SQLAlchemy pool can legitimately want ~375.

Reaping idle sessions was rejected: **18 of the 80** `create_engine`/`create_async_engine` calls in `orion/` and `services/` do not set `pool_pre_ping` (AST-counted, tests excluded), including live paths in `orion/substrate/mutation_queue.py` and `orion/substrate/policy_profiles.py`. Killing idle sessions would convert a capacity problem into scattered pool-checkout errors in exactly those pools.

## Why 300 and not 375

78 of the 80 non-test `create_engine` sites use SQLAlchemy's default sizing (5 + 10) and none set `poolclass`, so a simultaneous max-out across ~25 clients implies ~375. That has never happened — observed peak is ~100, and load fluctuates widely (55/100 during final verification, 94–100 earlier the same hour). 300 is ~3× observed peak. Sizing for a worst case that has never occurred would cost page cache to buy slots nothing has requested; the gate is what turns an approach to 300 into a warning rather than another silent outage.

## Memory

Backend anon RSS is ~2.6MB, so 300 backends ≈ 780MB. `oom_kill = 0`. **No availability risk.**

But the cgroup is already at **16.77 GB of its 17.18 GB limit (97.6%)** — `memory.current`, not `docker stats`, which subtracts `inactive_file`. That is almost entirely reclaimable page cache, and with `shared_buffers` at the 128MB default that page cache *is* this database's cache for a 40GB database. New backend memory comes out of it. The cost is cache pressure, not an OOM.

## Tests run

```text
ORION_TEST_POSTGRES_URI=... pytest tests/test_check_postgres_connection_headroom.py -q
29 passed in 0.19s        (27 unit + 2 live; the 2 skip cleanly without a database)
```

Mutation-tested against the real file, each mutation asserted to actually apply first:

```text
RED | client-backend filter neutered (OR 1=1)  | 2 failed, 27 passed
RED | free measured against the reserve        | 3 failed, 26 passed
RED | ordinary-role message dropped            | 1 failed, 28 passed
RED | unrelated error becomes an alarm         | 1 failed, 28 passed
RED | reserve warning never fires              | 2 failed, 27 passed
RED | superuser count read as zero             | 1 failed, 28 passed
RED | session no longer read-only              | 1 failed, 28 passed
RED | host default becomes docker-internal     | 1 failed, 28 passed
```

All ten static gates named in `.github/workflows/orion-static-gates.yml` pass (list derived from CI, not memory).

## Evals run

No eval harness exists for `services/orion-sql-db` (compose, env and hand-applied migration SQL only). This adds a repo-root script and its tests rather than service code. Not claiming eval coverage.

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-sql-db/.env \
  -f services/orion-sql-db/docker-compose.yml config
  27:      - max_connections=300
```

Zero-configuration live run from the host:

```text
postgres connections: 55/100 used (45 free, 45%) [reserved=3, superuser clients=55]
  WARNING: all 55 client backends are superusers, so the 3 reserved slots hold
  nothing back -- there is no emergency door for an operator when this fills.
  idle: 54 total, 5 idle >2h
```

## Review findings fixed

- **Finding: the 97 ceiling is wrong — services are superusers, so the wall is 100.**
  - Fix: `Headroom.free` now measures against `max_connections`; `nonsuperuser_ceiling` is reported separately.
  - Evidence: 217 `too many clients` vs 0 `remaining connection slots are reserved` in the server log; `pg_stat_activity` join on `pg_roles` shows 70/70 client backends are `postgres` with `usesuper = t`. Reproduced independently before acting.

- **Finding: `is_saturation_error` could not recognise the refusal an ordinary role receives.** Latent today; fires the moment anyone moves a service off the superuser — i.e. during the fix for the finding above.
  - Fix: added `remaining connection slots are reserved` to the message patterns.
  - Evidence: `test_the_ordinary_role_refusal_message_is_also_detected`; mutation "ordinary-role message dropped" → RED.

- **Finding: the SQL-text assertions were vacuous — `... OR 1=1` left the suite 22/22 green.**
  - Fix: replaced with a live test comparing `read_headroom().used` against `count(*) − count(background)` on a real server, skipped without a database.
  - Evidence: mutation "client-backend filter neutered (OR 1=1)" → RED (2 failed). It was GREEN before.

- **Finding: exit-code collision — alarm, missing DSN, bad password and missing driver all exited 1.**
  - Fix: `EXIT_ALARM=1` / `EXIT_CANNOT_CHECK=2`, matching `scripts/check_sql_migrations_applied.py`.
  - Evidence: `test_an_unrelated_connection_failure_is_not_laundered_into_an_alarm`; mutation "unrelated error becomes an alarm" → RED.

- **Finding: the advertised invocation could not alarm, twice over** — system `python3` has no psycopg2, and without `--gate` a saturated reading exits 0.
  - Fix: compose comment and Makefile both use `.venv/bin/python … --gate`.
  - Evidence: `python3 -c "import psycopg2"` → `ModuleNotFoundError`, venv import succeeds.

- **Finding: DSN resolution was a trap** — the root `.env` `POSTGRES_URI` is a docker-internal hostname that does not resolve from the host.
  - Fix: `connection_params()` adopts the sibling gate's `ORION_PG_*` convention defaulting to `localhost:55432`, so a bare invocation works.
  - Evidence: the zero-configuration run above; mutation "host default becomes docker-internal" → RED.

- **Finding: the comment's own demand estimate (~375) exceeded the ceiling it chose (297).**
  - Fix: reconciled explicitly in the compose comment and in "Why 300 and not 375" above.

- **Finding: the memory framing was backwards** — "affordable, especially with `shared_buffers` at the 128MB default". Low `shared_buffers` means page cache *is* the cache.
  - Fix: rewritten with `memory.current` (16.77/17.18 GB, 97.6%) and the ~2.6MB/backend measurement; reclassified from availability risk to cache pressure.

- **Finding (my own, added then removed): the semaphore pre-flight was vacuous.** It claimed PG allocates SysV semaphores proportional to `max_connections` and checked `SEMMNI`.
  - Fix: section deleted.
  - Evidence: `ipcs -s` inside the container returns **zero** semaphore arrays with 96 backends attached — PG on Linux uses unnamed POSIX semaphores. The check could not have failed regardless of host limits.

- **Finding: a `pool_pre_ping` count cited from a line-based grep was wrong** ("only 38").
  - Fix: AST-counted — 62 with, 18 without. Corrected in commit `d04d6acd6`, the compose comment and this report. The decision is unchanged; the evidence for it was wrong.

- **Finding: cron'ing the gate before the restart would arrive permanently red.**
  - Not fixed in code by design: the Makefile target exists but nothing schedules it yet. Scheduling should follow the restart. Called out under Restart required.

## Restart required

`max_connections` is postmaster-only. This does **not** take effect until orion-sql-db is fully restarted, and that restarts the database every Orion service depends on. Not run here — production, and Juniper's call.

```bash
cd /mnt/scripts/Orion-Sapienform
docker compose --env-file .env --env-file services/orion-sql-db/.env \
  -f services/orion-sql-db/docker-compose.yml up -d --force-recreate orion-sql-db
```

Then confirm the ceiling moved, and only then schedule the gate:

```bash
make postgres-headroom            # expect [reserved=3 ...] against 300
# */10 * * * * make postgres-headroom
```

Before restarting, check no backup or long migration holds a lock — a prior incident had boot sit 35 minutes at "Waiting for application startup" behind a backup.

## Risks / concerns

- **Severity: medium.** Restarting Postgres drops every dependent service's connections. The 62 pools with `pool_pre_ping` recover; the 18 without may need their own restart.
- **Severity: medium (not fixed here).** Services run as the `postgres` superuser, so there is no reserved slot for an operator during an incident. Raising the ceiling reduces how often that matters without addressing it. Follow-up: move services to a non-superuser role — at which point the ordinary-role refusal path (already handled) starts firing.
- **Severity: low.** 300 relieves the symptom, not the demand. If usage keeps climbing, the gate is what makes the next approach visible.
- **Observed, not fixed:** `shared_buffers` is at the 128MB default on a 16GB container holding a 40GB database, with 209,020 reclaim events and 78M workingset refaults. Almost certainly too small, but it is a separate performance decision needing its own evidence.
- **Disclosure:** some fraction of the 71 refusals logged in the 04:59/05:01 bursts came from this investigation's own connections (roughly a dozen `psql` calls plus a containerised smoke) against a database that had 3 slots free. `log_connections=off`, so they cannot be disentangled from real service refusals. 146 of the 217 are independent of any of this work — 28 predate the investigation entirely. The operational point stands on its own: diagnosing this database costs slots, which is an argument for fixing it before probing it further.

## Status

DONE_WITH_CONCERNS — code and instrument complete and verified; the ceiling is not live until orion-sql-db is restarted, and the superuser-role hazard is surfaced but not fixed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2010
