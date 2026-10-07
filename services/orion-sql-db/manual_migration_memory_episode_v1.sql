-- Memory episode redesign, Stage 1 PR 1 (2026-10-02).
-- Spec: docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md
--
-- Apply BEFORE deploying orion-memory-consolidation from this branch. The
-- service is fail-open without it (live window closing never depends on these
-- objects), but the shadow episodes and the close audit stay empty until it runs.
-- Additive only: no existing column changes meaning, nothing is dropped.

-- Why the live (legacy) rule closed each window, and the boundary score of the
-- turn that closed it. phase_change_at_close already exists and starts being
-- filled once the Hub stamps spark_meta.conversation_phase (Fix 1).
ALTER TABLE memory_consolidation_windows ADD COLUMN IF NOT EXISTS close_reason TEXT;
ALTER TABLE memory_consolidation_windows ADD COLUMN IF NOT EXISTS boundary_score_at_close DOUBLE PRECISION;

-- The GIN index for Fix 2's dedup lookup is built CONCURRENTLY in its own file (it cannot run inside
-- a transaction): manual_migration_memory_episode_v1_gin.sql. Rollback: *_rollback.sql.

-- Shadow episodes under boundary Rule 3. Read by nothing live.
CREATE TABLE IF NOT EXISTS memory_episode_shadow (
    episode_id TEXT PRIMARY KEY,
    source_platform TEXT,
    status TEXT NOT NULL DEFAULT 'open',          -- open | closed
    episode_status TEXT,                          -- closed | skipped (set at close)
    skip_reason TEXT,                             -- command_only
    boundary_rule TEXT NOT NULL DEFAULT 'v2',
    turns JSONB NOT NULL DEFAULT '[]',            -- per turn: phase, score, legacy + v2 decision
    started_at TIMESTAMPTZ NOT NULL,
    last_turn_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    closing_correlation_id TEXT,
    close_reason TEXT,
    phase_at_close TEXT,
    boundary_score_at_close DOUBLE PRECISION,
    close_lag_sec DOUBLE PRECISION,
    juniper_turn_count INTEGER,
    command_turn_count INTEGER,
    closed_event_published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- At most one open shadow episode per platform (NULL = direct conversation).
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_episode_shadow_open_platform
    ON memory_episode_shadow ((coalesce(source_platform, '')))
    WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_memory_episode_shadow_closed_at
    ON memory_episode_shadow (closed_at);
CREATE INDEX IF NOT EXISTS idx_memory_episode_shadow_unpublished
    ON memory_episode_shadow (closed_at)
    WHERE status = 'closed' AND closed_event_published_at IS NULL;
