CREATE TABLE IF NOT EXISTS memory_consolidation_windows (
    memory_window_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'open',
    turn_correlation_ids JSONB NOT NULL DEFAULT '[]',
    phase_change_at_close TEXT,
    consolidation_status TEXT,
    draft_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS memory_graph_suggest_drafts (
    draft_id TEXT PRIMARY KEY,
    memory_window_id TEXT NOT NULL REFERENCES memory_consolidation_windows(memory_window_id),
    status TEXT NOT NULL DEFAULT 'pending_review',
    draft JSONB NOT NULL,
    turn_correlation_ids JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (memory_window_id)
);

-- Partition the open-window cursor by originating platform.
--
-- _get_open_window() used to select the single global open window, so ai-town
-- NPC dialogue and real conversations with Juniper appended to the same window:
-- 26 windows were confirmed mixed on live data (2026-08-14). That is a
-- correctness bug in its own right -- an episode is supposed to be one coherent
-- stretch of conversation -- and it also capped the ai-town formation gate,
-- which can only treat a window as external when every turn in it agrees.
--
-- NULL means a direct hub/API conversation, which is what every window created
-- before this column carries and the correct default for them.
ALTER TABLE memory_consolidation_windows ADD COLUMN IF NOT EXISTS source_platform TEXT;

CREATE INDEX IF NOT EXISTS idx_memory_consolidation_windows_status ON memory_consolidation_windows(status);
-- The hot path is "the open window for this platform"; partial index because
-- only one row per platform is ever 'open'.
CREATE INDEX IF NOT EXISTS idx_mcw_open_by_platform
    ON memory_consolidation_windows (source_platform, created_at)
    WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_memory_graph_suggest_drafts_status ON memory_graph_suggest_drafts(status);
