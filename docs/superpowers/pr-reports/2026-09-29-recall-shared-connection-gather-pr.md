## Summary

- Recall looked up chat turns in two tables at once (the main chat log and the AI Town mirror) over a single Postgres connection. A single asyncpg connection can only run one query at a time, so the mirror query failed on every call and its rows were silently dropped.
- The two queries now run one after the other on the same connection. Each is still wrapped in its own try/except, so a failure in one table never throws away the other table's rows.
- Adds a regression test with a fake connection that behaves like asyncpg: it raises `InterfaceError` if a second query starts while the first is still running. The test fails on `main` and passes here.

## Outcome moved

- Live, 2026-09-29, over 24h: `orion-athena-recall` logged 1,075 `another operation is in progress` errors. 796 came from `fetch_chat_turns_by_id` and 279 from `fetch_chat_turn_timestamps`. Every one was on the **mirror** query; the primary query never failed.
- Effect: recall never returned an AI Town mirror row by id, and never used a mirror row's timestamp. Where an id existed in both tables, the primary row won instead of the mirror row, which is the opposite of the documented rule.
- After this patch that error count should be 0, and mirror rows should come back.

**The timeout hypothesis is not supported by the evidence.** The brief named this bug as the leading suspect for the `orion:exec:request:RecallService` RPC timeouts. The live data says otherwise:
- The bug fails fast. The second query errors immediately and adds no wait.
- Recall's own logs cover 274 requests since it restarted (2026-09-29 03:56 UTC). All 274 got a reply: median 0.55 s, p90 2.3 s, max 8.0 s. None took over 10 s, and 2 took over 5 s. For comparison, the caller's recall deadlines are 30 s and 90 s; the diagnostic recall deadline is 5 s.
- orion-equilibrium's `transport_baseline_obs` for the same period shows 305 successes and 2 timeouts. Both timeouts were at 03:57, during recall's own restart.
- The 09-28 14:28 peak is older than both containers' logs, so what caused it is **UNVERIFIED**.

## Current architecture

`services/orion-recall/app/sql_chat.py::_fetch_primary_and_mirror_rows` ran `asyncio.gather` over two `conn.fetch` calls on one `asyncpg.connect()` connection. The callers open one connection per call and close it afterwards. The earlier tests' fake connections allowed concurrent fetches, so they never caught this.

## Architecture touched

orion-recall only. There are no contract, bus, schema, or env changes. The patch still uses exactly one Postgres connection per call, so the connection budget is unchanged. Live check: Postgres was at 94 of 300 connections, and recall's short-lived connections did not show up among the top clients.

## Files changed

- `services/orion-recall/app/sql_chat.py`: the two queries now run in sequence with per-table error isolation. The docstring is updated and the unused `asyncio` import is removed.
- `services/orion-recall/tests/test_sql_chat_connection_exclusivity.py`: new regression test plus per-table isolation tests. One of these has the error happen while the connection is busy, then checks that the other query still runs on the same connection.
- `docs/superpowers/pr-reports/2026-09-29-recall-shared-connection-gather-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: mirror-table rows are now actually returned, and the mirror row wins when an id exists in both tables, as already documented.
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
# on main (fix stashed, new test present)
pytest services/orion-recall/tests -q
  5 failed, 277 passed. Includes both new *_on_one_connection tests, failing with
  InterfaceError "another operation is in progress" and missing mirror rows.

# on this branch
pytest services/orion-recall/tests -q
  3 failed, 279 passed

pytest tests/test_sql_chat_connection_exclusivity.py tests/test_sql_chat_fetch_by_id.py -q
  15 passed
```

The 3 remaining failures already fail on `main` without this patch and have nothing to do with it:
- `test_process_recall_active_turn_exclusion`
- `test_recall_policy_harness::...gating_suppression_and_selection`
- `test_recall_vector_amputation::...import_without_vector_adapter`

## Evals run

```text
None. orion-recall has no eval for sql_chat. The regression test plus the
post-deploy log count below serve as the behavior check.
```

## Docker/build/smoke checks

```text
Not deployed. Production deploys run from the primary checkout on main after merge.
```

## Review findings fixed

- Finding: the isolation test raised its error before the fake connection was marked busy, so it never exercised "an error on the connection, then the next query on the same connection".
  - Fix: the failing table now raises inside the busy section, and the busy flag is cleared in `finally`, which is how asyncpg behaves.
  - Evidence: 15 passed.
- Finding (nit): "the bug this helper fixes" in the docstring was ambiguous next to the new paragraph.
  - Fix: it now names the 2026-08-19 bug explicitly.
  - Evidence: diff.
- Also reviewed, no change needed: nothing else in orion-recall shares one asyncpg connection across concurrent tasks. `cards_adapter` acquires a connection from the pool for each call. The other `gather` sites run Falkor/redis queries through a thread-safe connection pool.

## Restart required

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-recall up -d --build
```

## Post-deploy checks

```bash
# should be 0 (it was 1,075 in 24h)
docker logs --since 1h orion-athena-recall 2>&1 | grep -c "another operation is in progress"

# should stay low; before this patch: 2 timeouts / 305 successes over roughly 24h, both during a restart
docker logs --since 24h orion-athena-equilibrium 2>&1 | grep transport_baseline_obs | grep RecallService | grep -v '"timeout_count": 0'
```

## Risks / concerns

- Severity: low
  - Concern: worst-case wall time is now two sequential round-trips instead of one.
  - Mitigation: the "parallel" version never really ran in parallel, because the second query always failed instantly. Each query is an indexed `= ANY($1)` lookup.
- Severity: medium (scope)
  - Concern: this fix is not expected to move RecallService RPC timeouts. The cause of the 09-28 reliability_pressure peak on that hop is UNVERIFIED.
  - Mitigation: run the equilibrium check above. If RecallService timeouts appear outside restarts, investigate them separately.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2396

🤖 Generated with [Claude Code](https://claude.com/claude-code)
