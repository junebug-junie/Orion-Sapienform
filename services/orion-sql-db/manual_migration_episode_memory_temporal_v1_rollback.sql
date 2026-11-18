-- DESTRUCTIVE
-- Rollback of manual_migration_episode_memory_temporal_v1.sql (issue #3659). Drops the eval and
-- run audit tables and the validity columns. Undo applied supersedes first
-- (scripts/memory_supersede_pass.py undo), and deploy orion-durable-runs from before #3659 part 2.
ALTER TABLE episode_memory DROP COLUMN IF EXISTS invalid_reason;
ALTER TABLE episode_memory DROP COLUMN IF EXISTS invalidated_by;
ALTER TABLE episode_memory DROP COLUMN IF EXISTS invalid_at;
DROP TABLE IF EXISTS memory_supersede_run;
DROP TABLE IF EXISTS memory_supersede_eval_run;
