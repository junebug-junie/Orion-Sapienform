-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_personality_graph_v1.sql
-- DESTRUCTIVE
-- Drops the personality graph tables. Set PERSONALITY_ENABLED=false on orion-durable-runs first.
BEGIN;
DROP TABLE IF EXISTS personality_weight_history;
DROP TABLE IF EXISTS personality_evidence;
DROP TABLE IF EXISTS personality_cursor;
DROP TABLE IF EXISTS personality_disposition;
COMMIT;
