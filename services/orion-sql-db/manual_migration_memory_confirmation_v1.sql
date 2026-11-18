-- Memory confirmation loop (2026-10-06, pulled forward from Stage 3 of
-- docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md).
--
-- Indexes only: the loop reuses existing tables -- orion_ask (the "Orion is asking" card,
-- manual_migration_walkway_camera_v1.sql), attention_loop_outcome (the one resolution record,
-- manual_migration_attention_loop_outcome.sql) and the episode_memory* shadow tables
-- (manual_migration_episode_memory_v1.sql). Apply after those three, BEFORE deploying the Hub
-- or orion-memory-consolidation from this branch. Additive; all three tables are small
-- (episode_memory 31 rows, attention_loop_outcome 314 rows on 2026-10-06), so plain CREATE INDEX.
-- Rollback: manual_migration_memory_confirmation_v1_rollback.sql.

-- The consumer's catch-up read: "which memory outcomes has no event applied yet".
CREATE INDEX IF NOT EXISTS idx_episode_memory_event_outcome
    ON episode_memory_event (outcome_id) WHERE outcome_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_attention_loop_outcome_memory_confirm
    ON attention_loop_outcome (created_at) WHERE loop_id LIKE 'memory-confirm-%';

-- The card opener ("oldest unasked high-stakes memory") and the expiry join on the loop id.
CREATE INDEX IF NOT EXISTS idx_episode_memory_pending_confirmation
    ON episode_memory (created_at) WHERE stakes = 'high' AND confirmation_state = 'pending_confirmation';
CREATE INDEX IF NOT EXISTS idx_episode_memory_confirmation_loop
    ON episode_memory (confirmation_loop_id) WHERE confirmation_loop_id IS NOT NULL;
