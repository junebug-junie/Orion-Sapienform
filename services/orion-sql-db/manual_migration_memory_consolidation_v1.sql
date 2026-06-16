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

CREATE INDEX IF NOT EXISTS idx_memory_consolidation_windows_status ON memory_consolidation_windows(status);
CREATE INDEX IF NOT EXISTS idx_memory_graph_suggest_drafts_status ON memory_graph_suggest_drafts(status);
