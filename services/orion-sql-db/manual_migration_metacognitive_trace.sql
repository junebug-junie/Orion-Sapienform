-- Create durable storage for metacognitive reasoning traces.
CREATE TABLE IF NOT EXISTS orion_metacognitive_trace (
    trace_id TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    session_id TEXT NULL,
    message_id TEXT NULL,
    trace_role TEXT NOT NULL,
    trace_stage TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    token_count INT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orion_metacog_trace_corr
    ON orion_metacognitive_trace (correlation_id);

CREATE INDEX IF NOT EXISTS idx_orion_metacog_trace_session
    ON orion_metacognitive_trace (session_id);

CREATE INDEX IF NOT EXISTS idx_orion_metacog_trace_created_at
    ON orion_metacognitive_trace (created_at);
