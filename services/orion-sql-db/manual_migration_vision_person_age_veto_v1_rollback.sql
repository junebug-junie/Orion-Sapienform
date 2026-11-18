-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_vision_person_age_veto_v1.sql
-- DESTRUCTIVE
ALTER TABLE vision_person_attribution DROP COLUMN IF EXISTS age_veto;
