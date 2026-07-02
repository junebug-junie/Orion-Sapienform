-- Coalition dwell and hysteresis tracking (rung-3 workspace focus stability)
CREATE TABLE IF NOT EXISTS substrate_coalition_dwell_log (
    dwell_id TEXT PRIMARY KEY,
    generated_at TIMESTAMP NOT NULL,
    coalition_ids JSONB NOT NULL,  -- ["node:a", "node:b", ...]
    candidate_ticks INT DEFAULT 0,
    active BOOLEAN DEFAULT FALSE,
    dwell_ticks INT DEFAULT 0,
    salience_trend FLOAT DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_coalition_dwell_generated ON substrate_coalition_dwell_log(generated_at DESC);
