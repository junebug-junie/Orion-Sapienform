-- Rollback of manual_migration_memory_episode_v1.sql (+ _gin.sql). Stop orion-memory-consolidation's
-- shadow first (MEMORY_EPISODE_SHADOW_ENABLED=false) or accept logged, fail-open errors.
-- Drops the shadow episodes (shadow data only) and the two audit columns. Run outside a transaction.
DROP INDEX CONCURRENTLY IF EXISTS idx_mcw_turns_gin;
DROP TABLE IF EXISTS memory_episode_shadow;
ALTER TABLE memory_consolidation_windows DROP COLUMN IF EXISTS close_reason;
ALTER TABLE memory_consolidation_windows DROP COLUMN IF EXISTS boundary_score_at_close;
