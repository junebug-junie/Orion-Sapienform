-- ORION-MIGRATION-REQUIRED-BY: none retention works without it, the index only makes it cheap, see #3763
-- orion-sql-writer retention for gpu_pool_events ages rows by created_at
-- (grammar_truth.py _batch_delete_sql / _count_debt_sql: WHERE created_at < :cutoff),
-- but every secondary index was on generated_at. So each 60 s retention cycle read the
-- whole table twice. Live 2026-11-25: ~876k rows (30-day retention), ~150 ms parallel
-- seq scan per pass, all from cache.
--
-- Additive only. CONCURRENTLY: no write lock on the live table. Run outside a transaction.
-- If interrupted, drop the INVALID index and re-run.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gpu_pool_events_created ON gpu_pool_events (created_at);
