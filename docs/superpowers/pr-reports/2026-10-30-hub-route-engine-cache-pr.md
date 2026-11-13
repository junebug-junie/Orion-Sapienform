## Summary

- Nine read-only hub debug routes (`substrate_*_routes.py`) built a new database connection pool on every request. They now build it once per module and reuse it.
- Four consolidation endpoints reached Postgres through `orion/consolidation/repository.py`, which also built a fresh pool per call. They now pass the module's pool in.
- New live eval measures connections left open after real route calls. New tests fail if either leak comes back, and a gate refuses new per-call pools anywhere in hub scripts.
- The hub labels its connections `orion-hub` (`PGAPPNAME`). The Postgres headroom alarm now names the top 3 holders by that label.

Deliberately small. Not changed: sharing pools across modules, timeouts, transactions, asyncpg pools, the 4 other known per-call pools (allowlisted in the gate), any other service.

## Outcome moved

Live eval (`services/orion-hub/evals/run_route_engine_pool_eval.py`, run against the live database from the hub image):

| | pools built per 25 requests | connections left open |
|---|---|---|
| before | 22-25 per module | 16 (FAIL) |
| consolidation fix removed | 1 per module | 18 (FAIL) |
| after | 1 per module | 9, one per module (PASS) |

Context at time of writing: Postgres had 183 of 300 connections in use; the hub held 90 of them (seen only as the Docker gateway IP 172.18.0.1). No "too many clients" refusals in 14 days; the headroom watch log peaked at 284/300.

## Current architecture

- Each of the nine modules had `_engine()` returning `create_engine(uri, pool_pre_ping=True)`. Routes use it as `with _engine().connect() as conn:`. The abandoned engine kept its connection open until garbage collection.
- `substrate_consolidation_routes.py` called `load_recent_motifs(uri, ...)` etc. without `engine=`; `repository._engine_for()` then called `create_engine` per call.
- `services/orion-hub/scripts/pg_engine.py` already held one shared cached engine (`get_engine()`), used by the outreach tick.
- `scripts/check_postgres_connection_headroom.py` grouped holders by client IP only; the alarm card had no breakdown.

## Architecture touched

- orion-hub route modules, `pg_engine.py`, hub compose (`hub-app` only), the repo-level headroom script. No bus, schema, or API contract change.

## Files changed

- `services/orion-hub/scripts/pg_engine.py`: adds `module_engine(owner, uri)`, an `lru_cache`d engine per (module, uri). Keyed per module on purpose, so modules keep separate pools as before.
- `services/orion-hub/scripts/substrate_{attention,biometrics,consolidation,execution_dispatch,feedback,field,observability,policy,proposal}_routes.py`: `_engine()` returns `module_engine(__name__, uri)`. Same name and signature, so existing tests that patch `_engine` are unaffected.
- `services/orion-hub/scripts/substrate_consolidation_routes.py`: the four repository loaders get `engine=_engine()`.
- `services/orion-hub/tests/test_route_engine_reuse.py`: reuse per module; separate pools per module; consolidation endpoints pass the module engine; AST gate against new per-call `create_engine` in hub scripts.
- `services/orion-hub/evals/run_route_engine_pool_eval.py`: the live eval above.
- `services/orion-hub/docker-compose.yml`: `PGAPPNAME=orion-hub` on `hub-app`.
- `scripts/check_postgres_connection_headroom.py`: holders grouped by `application_name` (falling back to IP); the alarm card lists the top 3 holders, best effort.
- `tests/test_check_postgres_connection_headroom.py`: 3 tests for the holder line.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: each of the nine modules now has one bounded pool (SQLAlchemy default: 5 + 10 overflow, 30 s checkout timeout) instead of unlimited fresh pools. A 16th simultaneous request to one module would wait instead of opening a new connection. Real load is far below this.
- Compatibility notes: none.

## Env/config changes

- Added keys: `PGAPPNAME=orion-hub`, a fixed literal in the `hub-app` compose `environment:` block. It is a libpq label, not operator config, so it is not in any `.env_example`.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/test_route_engine_reuse.py                  15 passed
  with the old per-request _engine() restored:                              2 failed
  with the consolidation engine= arguments removed:                         4 failed
pytest <15 existing hub test files touching these modules / pg_engine>      354 passed
pytest tests/test_check_postgres_connection_headroom.py                     46 passed, 2 skipped
pytest services/orion-hub/tests (full suite, repo root)                    4451 passed, 146 skipped, 0 failed
pytest <hub tests touching consolidation / pg_engine>, after review fix      473 passed, 8 skipped
```

`test_check_async_routes_not_blocking.py::test_the_real_repo_is_clean` fails only when pytest runs from inside `services/orion-hub/`; it fails identically on main there and passes from the repo root (as CI runs it).

## Evals run

```text
python services/orion-hub/evals/run_route_engine_pool_eval.py   (live DB, hub image, host network)
  main code:                      22-25 engines/module, 16 held -> FAIL
  consolidation fix removed:      18 held -> FAIL
  this branch:                    1 engine/module, 9 held -> PASS
consolidation loaders, live, repository engine vs plain engine: motifs/expectations/schema/tensor outputs identical
```

## Docker/build/smoke checks

```text
docker compose config (hub) -> renders PGAPPNAME: orion-hub
psycopg2 in the hub image honours PGAPPNAME; asyncpg does not (its pools stay unlabelled)
scripts/check_postgres_connection_headroom.py --verbose against live DB -> new holder query OK
```

## Review findings fixed

Code review ran in a subagent.

- Finding (medium): four consolidation endpoints still built a pool per request through `orion/consolidation/repository._engine_for()`; the gate could not see it.
  - Fix: pass `engine=_engine()`; eval now calls the real endpoints.
  - Evidence: eval 18 held -> 9 held; 4 new tests fail without the fix; live loader output identical across engines.
- Noted, no change: per-module pool cap (5 + 10 overflow) replaces unlimited fresh pools; observability uses about 2 connections per request.
- Noted, no change: two threads racing a cold `lru_cache` can build one extra engine once per module.
- Noted, no change: the AST gate is a ratchet, not a proof. It does not follow calls into other files, and fails safe on some cached patterns.
- Noted, no change: `PGAPPNAME` also labels scripts run by `docker exec` inside the hub container.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build hub-app
```

Run from the clean primary `main` checkout after merge. The headroom script is run by cron from the repo, so it picks up the change on pull.

## Risks / concerns

- Severity: low
- Concern: the cached pools now keep up to about 9 idle connections open for the life of the hub process; `--verbose` headroom output counts them as "idle > 2h". That is expected, not a leak.
- Mitigation: the eval bounds it at one per module.

- Severity: low
- Concern: 4 other per-call pools remain (`chat_history_rehydrate`, `curiosity_hint`, `hub_presence`, `outreach_provenance`), plus shared `orion/` modules the gate cannot scan.
- Mitigation: allowlisted in the gate so the list can only shrink; follow-up if wanted.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2993

🤖 Generated with [Claude Code](https://claude.com/claude-code)
