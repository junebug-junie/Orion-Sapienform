-- Rollback for manual_migration_memory_confirmation_v1.sql (indexes only; no data is lost).
-- Turn the loop off first: MEMORY_CONFIRMATION_LOOP_ENABLED=false in orion-memory-consolidation and orion-hub.
DROP INDEX IF EXISTS idx_episode_memory_event_outcome;
DROP INDEX IF EXISTS idx_attention_loop_outcome_memory_confirm;
DROP INDEX IF EXISTS idx_episode_memory_pending_confirmation;
DROP INDEX IF EXISTS idx_episode_memory_confirmation_loop;
