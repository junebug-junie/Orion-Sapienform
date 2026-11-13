-- ORION-MIGRATION-REQUIRED-BY: none rollback of manual_migration_personality_naming_v1.sql
-- DESTRUCTIVE
-- Rollback of manual_migration_personality_naming_v1.sql. Drops the naming log and columns.
DROP TABLE IF EXISTS orion_disposition_naming;
ALTER TABLE personality_disposition
    DROP COLUMN IF EXISTS name, DROP COLUMN IF EXISTS name_sentence, DROP COLUMN IF EXISTS named_at,
    DROP COLUMN IF EXISTS naming_id, DROP COLUMN IF EXISTS orion_verdict, DROP COLUMN IF EXISTS merged_into,
    DROP COLUMN IF EXISTS name_grounding_rank, DROP COLUMN IF EXISTS name_grounding_margin,
    DROP COLUMN IF EXISTS last_shown_at, DROP COLUMN IF EXISTS shown_count,
    DROP COLUMN IF EXISTS verdict_evidence_count;
