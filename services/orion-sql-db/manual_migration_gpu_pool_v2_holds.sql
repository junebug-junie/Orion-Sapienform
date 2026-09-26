-- orion-gpu-pool stage 4.3: durable-run holds and the swap actuation engine.
-- Spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md
-- Additive only, safe on the live tables. Apply BEFORE deploying a 4.3 pool: it refuses to boot
-- without these columns (app/store.py check_schema). Each statement is its own transaction when
-- run with `psql -f` (autocommit), which CREATE INDEX CONCURRENTLY requires.
--
-- lock_timeout: ADD COLUMN takes an ACCESS EXCLUSIVE lock for an instant (no rewrite: every new
-- column is nullable or has a constant default). If the pool is mid-commit it waits at most 5s
-- and fails instead of queueing every lease RPC behind it; just re-run the file.
SET lock_timeout = '5s';

-- A call made under a durable run's hold (verb=attach). NOT parent_lease_id: that already means
-- "replayed from" (backfill lineage) in this table.
ALTER TABLE gpu_pool_leases ADD COLUMN IF NOT EXISTS hold_lease_id text;

-- The swap state machine, persisted so a restarted pool reconciles instead of re-issuing.
ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS swap_role text;
ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS swap_generation integer NOT NULL DEFAULT 0;
ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS swap_action jsonb;
ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS residency_until timestamptz;
ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS loaded_at timestamptz;
-- role -> per-slot context last seen on this card. An unloaded swap seat reports none, and the
-- load decision for a hold with min_ctx_tokens needs it; kept across pool restarts.
ALTER TABLE gpu_pool_cards ADD COLUMN IF NOT EXISTS seen_ctx jsonb;

-- Children of one hold (acceptance check 2 joins on it). Partial: almost every lease is not a child.
-- If this build fails (e.g. lock_timeout) it leaves an INVALID index that IF NOT EXISTS then skips on
-- every re-run. Check: SELECT indisvalid FROM pg_index WHERE indexrelid='gpu_pool_leases_hold_idx'::regclass;
-- Recover: DROP INDEX CONCURRENTLY IF EXISTS gpu_pool_leases_hold_idx; then re-run this file.
CREATE INDEX CONCURRENTLY IF NOT EXISTS gpu_pool_leases_hold_idx ON gpu_pool_leases (hold_lease_id)
    WHERE hold_lease_id IS NOT NULL;
