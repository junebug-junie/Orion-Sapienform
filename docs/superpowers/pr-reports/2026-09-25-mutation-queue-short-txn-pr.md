## Summary

- The self-modification worker no longer keeps a Postgres transaction open for its whole cycle. Its "only one worker at a time" lock now uses a connection that never opens a transaction (AUTOCOMMIT), so the lock itself works exactly as before.
- Saving the mutation store now writes only the rows that changed since the last save, instead of re-saving all ~550k signals plus every other row in one giant transaction.
- Saves go out in short transactions of at most 500 rows each, through one engine the store reuses. Before, every save built a new engine.
- Loading at boot reads the raw rows inside a short transaction and does the slow parsing after that transaction has closed.
- Saves from the worker thread and from the hub's event loop now take turns (a store-level I/O lock), so an older version of a row can never overwrite a newer one and then be treated as current.
- Fixed a broken test helper in `test_mutation_store_incremental_persist.py` that had been failing 5 tests on main since the `routing` surface was retired.

## Outcome moved

The 2026-09-25 incident: two sessions from this worker sat "idle in transaction" for 7+ minutes. Any `CREATE INDEX CONCURRENTLY` in the database waited on them, which hung orion-gpu-pool's boot (LangGraph `saver.setup()`) for about 10 minutes. That was a full LLM outage until the sessions were terminated by hand.

After this patch:
- The lock connection shows `idle`, not `idle in transaction`. This was checked against a real throwaway Postgres 16.
- A save after one change writes 1 row in 1 short transaction. The old code wrote every row.
- A backlog is split into ≤500-row transactions.

## Current architecture

- `orion/substrate/mutation_worker.py::_acquire_leader_lock` called `engine.connect()` and then ran `SELECT pg_try_advisory_lock(914257)`. SQLAlchemy 2 automatically opened a transaction there, and it stayed open for the whole cycle.
- `orion/substrate/mutation_queue.py::_persist_to_postgres` ran one `engine.begin()` and executed one UPSERT per row across all ~19 tables on every `_persist()` call (24 call sites), with a fresh `create_engine` each time. The sqlite mirror (`_persist_to_sql`) did the same.
- The only production user is the hub singleton `SUBSTRATE_MUTATION_STORE` (`services/orion-hub/scripts/api_routes.py:534`), and the worker runs inside the hub.

## Architecture touched

These changes stay inside `orion/substrate/` (the store and the worker). Nothing changed in the bus, the schemas, the env, or compose.

## Files changed

- `orion/substrate/mutation_worker.py`: the leader-lock connection is now AUTOCOMMIT, and its engine uses NullPool so that `close()` really disconnects.
- `orion/substrate/mutation_queue.py`:
  - One table spec (`_ROW_TABLES`) drives both the save and the load.
  - `_dirty_write_ops` compares each row's payload digest with what each backend is known to hold. Signals use an append-only bookmark: the list object plus how many entries are already saved.
  - `_chunk_ops` packs rows into ≤500-row transactions, and each transaction runs as an executemany.
  - Adds the cached `_pg_engine()`, `_persist_io_lock`, a load that parses outside the transaction, and state seeding after a load.
- `orion/substrate/tests/test_mutation_store_short_transactions.py` (new): 12 regression tests. They run against a fake Postgres that records every transaction, a sqlite round trip, and the lock check.
- `orion/substrate/tests/test_mutation_store_incremental_persist.py`: the `_proposal` helper now builds the proposal directly, because the `routing` surface was retired and the old helper returned None. One comment was updated.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the table contents are identical. The one known difference is that rows already in the database are no longer rewritten when they haven't changed. Before, the first save after a boot rewrote everything, which also re-serialized old rows into the current model's shape (new default fields filled in). Now an unchanged old signal row keeps its stored JSON until it changes. Loading still validates every row through the model, so readers see the same objects either way.
- Compatibility notes: the table structure is unchanged, and no rows are deleted.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no template changed
- skipped keys requiring operator action: none

## Tests run

```text
# new regression tests: 12 passed
pytest orion/substrate/tests/test_mutation_store_short_transactions.py -q  -> 12 passed

# same tests against main's code: all 9 original tests FAIL for the claimed reason
#   (e.g. {'substrate_mutation_signal': 300} != {'substrate_mutation_proposal': 1};
#    one 1200-row transaction; engine created 8x; driver isolation None != 'AUTOCOMMIT')
# mutation checks on the 3 tests added after review (each fails with its fix removed):
#   JSONB normalization disabled -> test_postgres_round_trip_and_reload_does_not_rewrite[True] FAILS
#   io lock replaced by nullcontext -> test_concurrent_saves_are_serialized FAILS

pytest orion/substrate/tests -q  -> 823 passed, 3 failed
#   the 3 failures are test_felt_state_self_definition_lane.py and fail identically on main (unrelated)
#   5 test_mutation_store_incremental_persist failures on main are now fixed

services/orion-hub/tests (all 7 files that touch the store):
#   manual_route_routing 7 passed, scheduler_runtime 19 passed, self_modification_panel 9 passed,
#   signal_intake 14 passed, recall_canary_profile_seed 4 passed
#   substrate_review_runtime_hub_debug 2 failed / recall_strategy_profiles_runtime 5 failed:
#   the same tests fail on main (pre-existing, unrelated: app.js tab / canary endpoint tests)

static gates from .github/workflows/orion-static-gates.yml: all PASS; git diff --check clean
```

## Evals run

```text
No eval harness exists for orion/substrate mutation persistence. The periodic check that matters here
is a live one after deploy: pg_stat_activity must show no hub session sitting 'idle in transaction'
(see Restart required).
```

## Docker/build/smoke checks

```text
Throwaway postgres:16-alpine on 127.0.0.1:55499 (not production; removed afterwards):
  first persist (3000 signals, 200 proposals + queue): 1.09 s, source=postgres
  after 1 change:  [INSERT substrate_mutation_queue x1, DELETE active_surface x1]
  reload equal: proposals True, queue True, 3000 signals
  first persist after reload (real JSONB ::text round trip): [DELETE active_surface x1]  -- zero data rows
  leader lock held -> pg_stat_activity state: 'idle'  (was 'idle in transaction'); pg_locks advisory: 1
  after release -> pg_locks advisory: 0

Scale (in-memory, 550k signals + 2345 proposals + 2345 queue rows):
  steady-state dirty diff after one change: 0.15 s CPU, 1 row written
No docker compose build/up was run (per task: no deploy).
```

## Review findings fixed

- Finding (should): two overlapping saves (worker thread and hub event loop) could commit an older row version last while recording the newer digest. After that, the row would never be rewritten, so the database would keep stale data until the next change.
  - Fix: `_persist_io_lock` (an RLock, always taken outside `_lock`) wraps `_persist()` and every single-row helper. The single-row helpers now *forget* the row's digest instead of recording one, so at worst one row gets rewritten on the next save.
  - Evidence: `test_concurrent_saves_are_serialized` fails when the lock is swapped for `nullcontext`.
- Finding (nit): the fake Postgres returned the exact string that was written, so the reload test didn't exercise JSONB key reordering.
  - Fix: a parametrized variant makes the fake reorder keys and spacing.
  - Evidence: it fails when `_normalize_payload_text` is disabled.
- Finding (nit): changing `postgres_url` kept the old database's save state.
  - Fix: the known state is cleared when the engine is rebuilt.
  - Evidence: `test_postgres_url_change_resets_known_state`.
- Finding (nit): a crash partway through a save can leave the tables partly written.
  - Fix: none; this is recorded in Risks below.
- Finding (nit): the first sqlite fallback save commits once per batch.
  - Fix: none; it is only slower and changes nothing in the tables.

## Restart required

orion-hub bakes `orion/` into its image, so it needs a rebuild. After merge, from a worktree synced to main:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
# verify: no hub session idle in transaction for more than a few seconds
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select pid, state, now()-xact_start, left(query,60) from pg_stat_activity where state like 'idle in transaction%'"
```

## Risks / concerns

- Severity: low
  - Concern: a large backlog save now spans several transactions. If the process dies between two of them, the database can briefly hold, for example, a decision without the proposal state written after it.
  - Mitigation: the next save writes whatever is still missing. Active surfaces are rebuilt from adoptions on load. The common `add_proposal` path still writes the proposal and its queue item in one transaction.
- Severity: low
  - Concern: the in-memory `_signals` list is still unbounded (~550k objects loaded at boot). Saves no longer grow with its size, but memory and boot time still do.
  - Mitigation: follow-up. Its readers (`lifecycle_for_proposal`, `recent_signals`) only need recent signals plus the ones a proposal references, so the list could be bounded without deleting any table rows.
- Severity: low
  - Concern: the steady-state save still serializes every non-signal row (~9.4k live) to compute digests, about 0.3 s of CPU per save.
  - Mitigation: acceptable for now. If it becomes hot, the diff can be keyed on object identity.
- Status: UNVERIFIED in production until the hub is rebuilt and the `pg_stat_activity` check above has been run.

## PR link

(filled after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
