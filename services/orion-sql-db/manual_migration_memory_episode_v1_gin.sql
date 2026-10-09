-- Memory episode Stage 1 (PR #2479 follow-up, 2026-10-02): the Fix 2 dedup lookup index.
-- CONCURRENTLY so building it never blocks consolidation's writes to the live table.
-- Must run OUTSIDE a transaction:  psql -d conjourney -f manual_migration_memory_episode_v1_gin.sql
-- (plain psql -f autocommits each statement; do NOT pass --single-transaction / -1).
-- Optional for correctness: without it the lookup is a sequential scan of a small table.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mcw_turns_gin
    ON memory_consolidation_windows USING GIN (turn_correlation_ids jsonb_path_ops);
