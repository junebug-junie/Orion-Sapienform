-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_personality_choice_v1.sql
-- DESTRUCTIVE
-- Rollback of manual_migration_personality_choice_v1.sql. Stop orion-durable-runs' personality loop
-- first (PERSONALITY_ENABLED=false). Drops every recorded choice.
BEGIN;
ALTER TABLE personality_disposition DROP COLUMN IF EXISTS opposed_count;
DROP TABLE IF EXISTS personality_choice;
COMMIT;
