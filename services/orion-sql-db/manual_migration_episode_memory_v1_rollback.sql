-- Rollback of manual_migration_episode_memory_v1.sql. Shadow data only: nothing live reads these.
-- Set MEMORY_EPISODE_WRITER_ENABLED=false on orion-durable-runs first.
DROP TABLE IF EXISTS episode_distill_run;
DROP TABLE IF EXISTS memory_tension_shadow;
DROP TABLE IF EXISTS episode_memory_event;
DROP TABLE IF EXISTS episode_memory_referent;
DROP TABLE IF EXISTS episode_memory_evidence;
DROP TABLE IF EXISTS episode_memory;
