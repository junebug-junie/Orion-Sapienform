-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_physiology_state_v1.sql
-- DESTRUCTIVE
-- Rollback for body economy Patch A3: drops the physiology history. Apply by hand only.
DROP TABLE IF EXISTS physiology_state;
