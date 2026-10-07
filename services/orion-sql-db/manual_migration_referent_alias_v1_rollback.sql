-- Rollback of manual_migration_referent_alias_v1.sql. Set MEMORY_REFERENTS_ENABLED=false and
-- MEMORY_REFERENT_PROJECTOR_ENABLED=false first. Falkor projections stay; retract by producer.
DROP TABLE IF EXISTS referent_projection;
DROP INDEX IF EXISTS idx_episode_memory_referent_node;
ALTER TABLE episode_memory_referent DROP COLUMN IF EXISTS node_id;
DROP TABLE IF EXISTS referent_alias;
