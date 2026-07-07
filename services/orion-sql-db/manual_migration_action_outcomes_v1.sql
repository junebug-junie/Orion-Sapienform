-- Action outcome durable store (Option C: bus event -> sql-writer -> SQL read).
-- Producer: orion-spark-concept-induction (action.outcome.emit.v1)
-- Consumer: orion-cortex-exec chat stance (load_action_outcomes)
--
-- sql-writer applies this automatically on boot via app/main.py lifespan
-- (CREATE TABLE / CREATE INDEX IF NOT EXISTS). This file is the equivalent
-- standalone DDL for manual application against a running Postgres.

CREATE TABLE IF NOT EXISTS action_outcomes (
    action_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    success BOOLEAN NULL,
    surprise DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    observed_at TIMESTAMPTZ NULL,
    correlation_id TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_action_outcomes_subject ON action_outcomes (subject);
CREATE INDEX IF NOT EXISTS idx_action_outcomes_subject_observed_at ON action_outcomes (subject, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_outcomes_correlation_id ON action_outcomes (correlation_id);
