# GPU pool: lease checkpoints in their own schema, bounded retention

## Summary

- The pool's LangGraph checkpoint tables move into their own Postgres schema, `gpu_pool`. They no longer share `public.checkpoints` with durable-runs.
- On boot, the pool moves any lease threads it left in the shared tables into its own schema. The move is idempotent and runs only after the leader lock, so the previous process has exited.
- A lease that ended (released or unavailable) is forgotten after 7 days (`GPU_POOL_LEASE_RETENTION_HOURS`). Both its checkpoint history and its `gpu_pool_leases` row are deleted, in batches of 2,000. Both tables stay about one retention window deep. Dead letters are kept for operator replay. The Hub's historical views read `gpu_pool_events`, which has 30-day retention, so they are unaffected.
- Backfill now reports the leases it can't replay because their history was pruned (`skipped_no_history`) instead of silently dropping them.

## Outcome moved

Found live about 25 minutes after stage 3 (#2328) went out:

- **Slow pool answers.** The pool took a median of 280 ms to answer a lease request. The slowest 1% took 4.8 s, and the maximum was 9.2 s.
- **Orphaned leases.** Nine acquire requests timed out. Each one left a lease the pool had granted but nobody held, until it expired 30 seconds later. That was 9 of the 9 expired leases.
- **Blocked fast lane.** At 04:04 three of these orphans held 3 of the fast lane's 4 slots at once.
- **The cause.** Every 2 minutes durable-runs' resume sweep reads *every* row of `public.checkpoints`, with all its stored data (`alist(None)`, averaging 513 ms and peaking at 8.5 s). The pool's lease threads had become most of that table: 857 threads and 4,410 checkpoints in 25 minutes, growing by about 1.5 GB a day. The pool's slow replies cluster around each run of that scan.

After this PR, durable-runs' scan only sees its own runs again, and the pool's own tables stay about 7 days deep.

## Current architecture

The pool ran one LangGraph thread per lease (`gpu_pool:<lease_id>`), using `AsyncPostgresSaver` on the same database and the same `public` tables as durable-runs. Nothing ever deleted a lease's checkpoints.

## Architecture touched

- `services/orion-gpu-pool` only:
  - connection settings
  - boot sequence
  - a background prune task
  - backfill reply detail
- No bus, schema-registry or API contract change. `GpuPoolControlReplyV1.detail` is a free-form dict.

## Files changed

- `services/orion-gpu-pool/app/store.py`: `pool_kwargs()` (search_path), `ensure_checkpoint_schema()`, `adopt_public_checkpoints()`, `prune_checkpoints()`.
- `services/orion-gpu-pool/app/main.py`: uses the above; `_prune_forever` runs every 10 minutes, outside the runtime lock.
- `services/orion-gpu-pool/app/runtime.py`: backfill reports `skipped_no_history`.
- `services/orion-gpu-pool/app/settings.py`, `.env_example`: `GPU_POOL_LEASE_RETENTION_HOURS=168`.
- `services/orion-gpu-pool/README.md`: why the schema is separate.
- `services/orion-gpu-pool/tests/test_store_postgres.py`: the production connection setup, plus 3 new Postgres tests.

## Schema / bus / API changes

- Added: Postgres schema `gpu_pool`, holding `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` and `checkpoint_migrations`. The pool creates it itself on boot.
- Behavior changed: the backfill reply adds `skipped_no_history`.
- Compatibility: lease threads already in `public` are moved on first boot. No manual migration is needed.

## Env/config changes

- Added keys: `GPU_POOL_LEASE_RETENTION_HOURS` (default 168). It never shipped under the draft name `GPU_POOL_CHECKPOINT_RETENTION_HOURS`, and that key was removed from the local `.env`.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The key is present in the primary `services/orion-gpu-pool/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-gpu-pool tests against a throwaway postgres:16: 40 passed (the 5 Postgres tests ran; they were skipped before)
Mutation check: with search_path back to public, the 3 new tests fail; restored, they pass
orion/gpu_pool + env sync tests: 86 passed
```

## Evals run

```text
No eval change: this is storage placement and retention, not scheduling. The scheduler eval is unaffected.
```

## Docker/build/smoke checks

Rehearsed on a copy of production's real rows: 4,620 checkpoints, 35k writes, and one durable-runs thread as a control. The rehearsal used a scratch Postgres 16 and the pool's real boot sequence.

```text
adopted rows 44354 in 1.14s; second pass 0
public.checkpoints: 0 pool threads left; the durable-runs thread is untouched
history of an adopted released lease reads back: ['admit', 'grant', 'expire']
prune (cutoff=now): 4611 checkpoints in 0.13s
```

Live deploy: see "Restart required".

## Review findings fixed

- Finding: the prune re-scanned all lease history every 10 minutes. `gpu_pool_leases` had no retention and no index on released + updated_at, so the cost grew forever: the same kind of stall this PR removes.
  - Fix: the lease row is deleted together with its thread, in `FOR UPDATE SKIP LOCKED` batches of 2,000. Each pass now only sees what aged out since the last one.
  - Evidence: `test_prune_forgets_only_old_ended_leases_in_bounded_batches` has 4 leases age out with batch=2, then a second pass forgets 0.
- Finding: leases that end `unavailable` (queue deadline, no card) are terminal but were never pruned. They would build up during an outage.
  - Fix: `PRUNABLE_STATUSES = ("released", "unavailable")`.
  - Evidence: the same test. With `unavailable` removed from the tuple, that test fails.
- Finding: the adopt step's `SET lock_timeout` stayed on a pooled autocommit connection for its next user.
  - Fix: `SET LOCAL` inside each per-table transaction.
  - Evidence: the adopt test checks `SHOW lock_timeout` on every pooled connection. With a plain `SET`, that test fails.
- Finding: the tests never reproduced production's layout (the projection in public, and durable-runs' checkpoint tables already there at migration v9 before the pool's `setup()`).
  - Fix: `_pool()` builds exactly that layout on a plain connection first. The isolation test asserts the projection is in public, `public.checkpoints` is untouched, and the pool schema's migrations are at the same version.
- Not bugs (checked by the reviewer):
  - `setup()` DDL resolves only to the first schema on the search_path.
  - psycopg accepts `options` in the pool's connection settings (direct Postgres, no pgbouncer).
  - The leader connection doesn't need the search_path.
  - The adopt step is idempotent and crash-safe.
  - Deleting from public doesn't conflict with durable-runs.
  - Nothing else reads the pool's checkpoints.

## Restart required

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

The boot log should show `gpu_pool_checkpoints_adopted rows=...` once. After that:

```sql
select count(*) from public.checkpoints where thread_id like 'gpu_pool:%';   -- 0
```

## Risks / concerns

- Severity: medium (rollback only).
  - Concern: rolling back is one-way. After the new pool adopts its threads, the old image reads `public.checkpoints` and finds no history for leases that were live at deploy time.
  - Mitigation: those leases last seconds to minutes. For a full rollback, move the rows back by hand:

    ```sql
    INSERT INTO public.<t> SELECT * FROM gpu_pool.<t> ON CONFLICT DO NOTHING
    ```

    Run it for each of checkpoints, checkpoint_blobs and checkpoint_writes.

- Severity: low.
  - Concern: backfill and the Hub history walker can only reach 7 days back.
  - Mitigation: the reach is configurable, and backfill names every lease it skips.
- Severity: low.
  - Concern: durable-runs' sweep is still an unfiltered full scan of its own table, so it grows with durable-runs' own history.
  - Mitigation: that's a separate durable-runs problem, out of scope here.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
