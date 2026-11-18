-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_personality_juniper_verdict_v1.sql
-- DESTRUCTIVE
-- Rollback of manual_migration_personality_juniper_verdict_v1.sql. Drops every recorded answer.
-- Tendencies already retired by a "no" stay retired (retired is terminal).
BEGIN;
DROP TABLE IF EXISTS personality_juniper_verdict;
COMMIT;
