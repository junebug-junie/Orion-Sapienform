-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_person_merge_v1.sql
-- DESTRUCTIVE
ALTER TABLE person DROP COLUMN IF EXISTS merged_into;
