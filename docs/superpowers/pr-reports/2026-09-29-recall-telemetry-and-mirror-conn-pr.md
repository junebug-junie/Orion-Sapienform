## Summary

- Recall now saves a timing row for every request. The table meant for this (`recall_telemetry`) had never saved a single row: every insert failed on a Python dict the database driver can't convert, and the error was logged at debug level so nobody saw it.
- That write no longer blocks the service. It used to open a database connection synchronously on the event loop for every recall, which stalled every other in-flight request while it ran. It now runs on a worker thread.
- Recall now actually returns AI-Town chat turns. The main and AI-Town chat queries ran at the same time on one database connection, which only allows one query at a time, so the AI-Town query failed on every call (724 warnings/hour live).

## Outcome moved

- The next recall timeout burst can be explained. On 09-28, 14:07–14:28 UTC, 26 recall calls timed out after 90s, and nothing recorded recall's own timings, so the cause is still unknown (UNVERIFIED: it lines up with the #2381/#2384/#2385 deploys and GPU-lease timeouts).
- Recall results can include AI-Town history again.

## Current architecture

- `services/orion-recall/app/worker.py::_persist_decision`: sync psycopg2, new connection + `CREATE TABLE IF NOT EXISTS` + insert per request, called directly from async handlers (bus handler and `/recall`).
- `services/orion-recall/app/sql_chat.py::_fetch_primary_and_mirror_rows`: `asyncio.gather` of two `conn.fetch` calls on one asyncpg connection.

## Architecture touched

orion-recall only. No bus, schema, or env changes.

## Files changed

- `services/orion-recall/app/sql_chat.py`: sequential primary/mirror fetch, each still isolated in its own try/except.
- `services/orion-recall/app/worker.py`: `Json()` for jsonb params, create-table once per process, warn on first failure, new `persist_decision_async` (`asyncio.to_thread`).
- `services/orion-recall/app/main.py`: `/recall` awaits `persist_decision_async`.
- `services/orion-recall/tests/test_sql_chat_fetch_by_id.py`: regression test with a one-statement-at-a-time fake connection.
- `services/orion-recall/tests/test_recall_telemetry_persist.py`: new; binds params through psycopg2's real adapter, checks create-once, off-loop thread, warn-once.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `recall_telemetry` receives rows (~1 per recall request); mirror rows are returned.
- Compatibility notes: table already exists live with the matching shape.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-recall/tests -q
  before (origin/main code + new tests): 7 failed, 276 passed
  after:                                 3 failed, 281 passed
  3 remaining failures are pre-existing on main and unrelated:
    test_process_recall_active_turn_exclusion (fake _query lacks `lane` kwarg)
    test_recall_policy_harness diagnostic
    test_recall_vector_amputation import
New mirror regression test fails on the old sql_chat.py, passes on the new one.
```

## Evals run

```text
None. orion-recall has no eval for telemetry persistence or the chat-mirror join.
Follow-up: once deployed, recall_telemetry itself is the eval surface (latency_ms distribution).
```

## Docker/build/smoke checks

```text
Live, in orion-athena-recall, rolled back (no row kept):
  raw dict insert  -> ProgrammingError: can't adapt type 'dict'   (reproduces the bug)
  Json() insert    -> LIVE ROW: [(['a'], {'vector': 3}, 812)]; count after rollback: 0
Image not rebuilt/deployed from this branch: post-deploy row flow UNVERIFIED.
```

## Review findings fixed

Code review subagent: no material findings. It confirmed failure isolation still holds sequentially, that nothing else calls `_persist_decision`, that `Json()` is correct for both columns, and that the tests would catch the original bugs.

- Finding (non-blocking): the bus handler awaits the telemetry write before replying, and `psycopg2.connect` had no timeout, so a hung DB connect would hang that recall's reply.
  - Fix: `connect_timeout=3` on the telemetry connection.
  - Evidence: `test_connect_is_time_bounded`.
- Finding (non-blocking, not fixed): if `recall_telemetry` is dropped while the service runs, later inserts fail at debug level until restart. Judged not worth handling.

## Restart required

```bash
scripts/safe_docker_build.sh orion-recall up -d --build
# verify:
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), max(created_at) from recall_telemetry"
docker logs --since 10m orion-athena-recall 2>&1 | grep -c mirror_query_failed   # expect 0
```

## Risks / concerns

- Severity: low
  - Concern: one extra Postgres connection per recall (short-lived, on a thread).
  - Mitigation: ~4 recalls/min live; a pooled connection is a follow-up if volume grows.
- Severity: low
  - Concern: `recall_telemetry` has no retention; it grows about 6k rows/day.
  - Mitigation: small rows; add pruning if it becomes material.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
