-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_person_record_v1.sql
-- DESTRUCTIVE
DROP TABLE IF EXISTS person_fact;
DROP TABLE IF EXISTS person_link;
DROP TABLE IF EXISTS person;
