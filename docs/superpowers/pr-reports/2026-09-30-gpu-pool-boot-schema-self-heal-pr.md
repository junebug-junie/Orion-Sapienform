## Summary

- The GPU pool now adds its own missing additive database columns when it boots, instead of refusing to start. Every LLM call leases through the pool, so a pool that won't start means all LLM calls fail. That happened twice: 2026-09-26 around 02:32-02:40, and 2026-09-30 09:01-09:09.
- If it can't get the table lock in time (for example a `pg_dump` or a long transaction holds it), the pool **does not exit**. It serves "degraded": the missing columns are kept in memory, it retries in the background, and it writes the held values into Postgres once the columns exist. The log shows this at CRITICAL level, and so does `/health` (`.schema`).
- Only additive columns are ever applied at boot. Tables, indexes, type changes and drops remain operator-only. A pool missing one of those still refuses to boot, loudly (`SchemaNotHealable`).
- A new drift gate test keeps the pool's boot list and the `manual_migration_gpu_pool_*.sql` files identical, column for column and in type/default.
- README, both SQL headers and two runbooks now say that running the migration first is "recommended", no longer "required". The manual files stay.

## Outcome moved

Failure mode removed: deploying a pool image before its additive migration no longer takes down every LLM call. The worst case is now a degraded pool that keeps serving and is visible in `/health`.

## Current architecture

`PostgresStore.check_schema()` (`services/orion-gpu-pool/app/store.py`) ran a `SELECT` of the v2/v3 columns before the leader lock was taken. If a column was missing the SELECT raised, lifespan failed and the container crash-looped. The gateway gets every LLM lease from the pool, so that was a total outage until someone ran the `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` by hand.

## Architecture touched

- `orion-gpu-pool` boot sequence, now: leader lock -> `heal_schema()` -> adopt checkpoints -> runtime start. The heal runs after the lock, so there is a single writer.
- The store's write and read paths gain a degraded-mode overlay: `_split`/`_merge`, `_degraded_write` with `_heal_lock`, and `_flush_overlay`.
- `/health` gains a `schema` block.
- No bus, schema-registry, env, config or launch-digest changes.

## Files changed

- `services/orion-gpu-pool/app/store.py`: `BOOT_ADDITIVE_COLUMNS` (the source of truth, mirroring v2/v3), `check_schema` (now returns the healable missing columns and raises `SchemaNotHealable` otherwise), `heal_schema`, `heal_forever`, the overlay, `schema_status`.
- `services/orion-gpu-pool/app/main.py`: the heal now runs after `leader()`, a background retry task starts when degraded, and `/health.schema` is added.
- `services/orion-gpu-pool/tests/test_schema_drift_gate.py` (new, needs no DB): the boot list and the migration files match in both directions; every column the store writes is created by v1 or by the boot list; the boot DDL is additive only.
- `services/orion-gpu-pool/tests/test_schema_self_heal_postgres.py` (new, real Postgres): self-heal on a v1-only DB, with the result matching the operator-migration schema; a missing table or v1 column still refuses to boot; lock contention leads to degraded serving (hold/attach idempotency, `seen_ctx`, emergency stop), then healing and write-through, and a restart sees the values; the reviewer's race is covered; retry latency under a dump lock is covered.
- `services/orion-gpu-pool/tests/test_health_schema.py` (new): `/health` reports a degraded schema.
- `services/orion-gpu-pool/tests/test_store_postgres.py`: the old "check_schema raises" assertions are replaced by "check_schema names what is owed".
- `services/orion-sql-db/manual_migration_gpu_pool_v2_holds.sql`, `..._v3_actuation_pause.sql`: header comments only. The SQL is unchanged.
- `services/orion-gpu-pool/README.md`, `docs/runbooks/2026-09-25-gpu-pool-stage4-cutover.md`, `docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md`: the "refuses to boot / apply first" wording is updated.
- `.github/workflows/orion-gpu-pool-tests.yml`: the push path filter now uses the glob `manual_migration_gpu_pool_*.sql`, so a new migration file triggers the drift gate.

## Schema / bus / API changes

- Added: `/health` field `schema` = `{state: ok|degraded|unchecked, missing, in_memory, applied, attempts, last_error, last_attempt_at, degraded_since, healed_at, overlay_rows}`.
- Removed: `V2_LEASE_COLUMNS`, `V2_CARD_COLUMNS`, `V3_CARD_COLUMNS` (replaced by `BOOT_ADDITIVE_COLUMNS`; they had no other users).
- Renamed: none.
- Behavior changed: the pool boots on a DB that is missing the v2/v3 columns. `check_schema()` returns a list instead of raising for those columns.
- Compatibility notes: the DDL is identical to the operator files, and the test asserts the resulting `information_schema` matches. Running the manual files afterwards is a no-op. The v2 partial index `gpu_pool_leases_hold_idx` is **not** built at boot, so still run v2 for it (it is a performance aid, not correctness).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

Timeouts are code constants (`HEAL_LOCK_TIMEOUT_MS=3000` at boot, `HEAL_RETRY_LOCK_TIMEOUT_MS=300`, `HEAL_STATEMENT_TIMEOUT_MS=15000`, backoff 5/10/30/60/120 s), not env keys, so they add no new config surface that could drift.

## Behaviour

| DB state at boot | Before | After |
|---|---|---|
| v2/v3 columns missing, lock free | crash-loop -> total LLM outage | columns added (log `gpu_pool_schema_healed`), `schema.state=ok`, serves normally |
| v2/v3 columns missing, table locked (dump/long txn) | crash-loop | serves **degraded**: CRITICAL `gpu_pool_schema_degraded` naming the columns and the migration file; `/health.schema.state=degraded`; missing values kept in memory; background retry; on success the values are written through (`gpu_pool_schema_recovered`) |
| v1 table or v1 column missing | crash-loop | crash-loop, with `SchemaNotHealable` naming the file (unchanged on purpose: the pool never creates tables) |

Why degraded-and-serving rather than waiting at boot: waiting keeps the outage going, because nothing else can grant a lease. A degraded pool only risks losing, if it restarts before healing, the in-memory values of the added columns (hold links, persisted swap state, the persisted pause). The runtime already carries those in memory. The emergency stop is never refused while degraded.

Lock hygiene: the heal runs one `ALTER TABLE public.<t> ADD COLUMN IF NOT EXISTS ..., ...` per table, and stops at the first lock failure. While serving, the retry waits at most 300 ms, and it checked that a racing lease write is delayed less than 1 s. This follows the memory lesson "boot DDL can hang behind a backup": nothing at boot waits more than 3 s per table.

## Tests run

```text
cd services/orion-gpu-pool && GPU_POOL_TEST_POSTGRES_URI=postgresql://postgres:t@127.0.0.1:55501/gpu_pool_test \
  PYTHONPATH=<worktree> python -m pytest tests -q -p no:cacheprovider
143 passed
(throwaway postgres:16 container on 127.0.0.1:55501, removed afterwards)
```

Mutation checks (each reverted afterwards):
- overlay write disabled -> the contention test fails (attach returns `unavailable`)
- flush replaced by a plain clear -> the contention test fails (values not persisted)
- `_heal_lock` removed -> the race test fails (heal completes mid-write)
- a v4 migration adding a column with no boot entry -> the drift gate fails
- a boot entry of `NOT NULL` without a default -> the drift gate fails (2 tests)

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py -> VERDICT: PASS
```

The eval uses MemoryStore, so it is unaffected by this change. It was run as a regression check. The heal behaviour is covered by the Postgres tests above, not by an eval.

## Docker/build/smoke checks

```text
Not deployed (per task). No Dockerfile/requirements/compose change. No live smoke run.
Live behaviour is UNVERIFIED until the next pool deploy. Check it with:
  curl -s localhost:8127/health | jq .schema    # expect state=ok, missing=[]
```

## Review findings fixed

- Finding (blocker): a flush running between a write's overlay step and its INSERT would UPDATE 0 rows and then clear the overlay, so a new child's `hold_lease_id` was lost. The reviewer reproduced this.
  - Fix: every degraded-mode write, and the flush, share one `asyncio.Lock` (`_degraded_write`). Once the pool is healthy, writes take the lock-free fast path. The flush is a single transaction.
  - Evidence: `test_heal_waits_for_a_degraded_write_in_flight_so_a_new_rows_value_is_not_lost`. It fails when the lock is removed.
- Finding (blocker): a write resetting a column to its default during the flush could leave a stale non-default value in the DB.
  - Fix: the same lock, since no write can interleave with the flush any more.
  - Evidence: same test, plus the lock mutation check.
- Finding (should): a degraded `set_actuation_paused` could write its overlay after the heal had cleared it, so the pause would not be persisted.
  - Fix: it now goes through `_degraded_write` and re-checks `_missing` inside the lock.
  - Evidence: the contention test asserts the pause is persisted after the heal and survives a restart.
- Finding (should): one ALTER per column meant about 9 x lock_timeout of queued writes per attempt.
  - Fix: one ALTER per table, stop at the first failure, retry timeout 300 ms.
  - Evidence: `test_a_background_retry_under_a_dump_stalls_lease_writes_briefly_not_per_column` (one `LockNotAvailable`, write wait < 1 s).
- Finding (should): the tests covered no interleavings.
  - Fix: the two tests above were added.
- Finding (nit): the ALTER was unqualified, while `search_path` puts `gpu_pool` first.
  - Fix: `ALTER TABLE public.<t>` and `UPDATE public.<t>`.
- Finding (nit, not changed): the overlay grows with the number of child leases for as long as the pool is degraded. It is bounded by the retention window (prune pops entries) and visible as `overlay_rows` in `/health`.
- Finding (nit, not changed): after a partial heal, `_missing` does not shrink until every column has healed. This is correct, just slightly conservative.

## Restart required

```bash
# from a worktree of merged main, when Juniper chooses to deploy:
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
curl -s localhost:8127/health | jq '{ok, schema}'
```

## Risks / concerns

- Severity: low
  - Concern: a degraded pool restarted before it heals loses the in-memory values of the added columns (hold links, swap state, the pause).
  - Mitigation: this is loud (CRITICAL log plus `/health`). The retry backoff caps at 120 s. The operator fix is to run the named migration.
- Severity: low
  - Concern: the pool's DB role must be allowed to ALTER the projection tables.
  - Mitigation: production connects as `postgres`. Without permission it degrades and keeps serving, rather than crashing.
- Severity: low
  - Concern: the `gpu_pool_leases_hold_idx` index is still operator-only.
  - Mitigation: it only affects performance. The README says to still run v2.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2438

🤖 Generated with [Claude Code](https://claude.com/claude-code)
