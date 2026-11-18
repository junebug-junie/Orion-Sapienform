-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_vision_office_session_reach_out_v1.sql
-- DESTRUCTIVE
ALTER TABLE vision_office_session DROP COLUMN IF EXISTS reach_out_decision_id;
ALTER TABLE vision_office_session DROP COLUMN IF EXISTS reached_out_at;
