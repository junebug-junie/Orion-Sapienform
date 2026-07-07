-- Action outcome durable store (Option C: bus event -> sql-writer -> SQL read).
-- Producer: orion-spark-concept-induction (action.outcome.emit.v1)
-- Consumer: orion-cortex-exec chat stance (load_action_outcomes)
--
-- On boot, sql-writer creates the table via Base.metadata.create_all (from the
-- ActionOutcomeSQL model) and creates these indexes via the equivalent DDL in
-- app/main.py lifespan (CREATE INDEX IF NOT EXISTS). This file is the standalone
-- equivalent for manual application against a running Postgres; the CREATE TABLE
-- here is a harmless no-op once the model has already created it.
--
-- Indexes are owned here (not via model index=True) to keep a single source of
-- truth: a composite (subject, observed_at DESC) matching the read query plus a
-- correlation_id lookup index. A standalone subject index is intentionally omitted
-- because the composite covers subject-prefix lookups.

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

CREATE INDEX IF NOT EXISTS idx_action_outcomes_subject_observed_at ON action_outcomes (subject, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_outcomes_correlation_id ON action_outcomes (correlation_id);
