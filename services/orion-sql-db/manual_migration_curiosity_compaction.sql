-- ORION-MIGRATION-REQUIRED-BY: orion-dream, orion-durable-runs
-- Additive checkpoint/artifact storage. No canonical memory/source mutations.
CREATE TABLE IF NOT EXISTS curiosity_compaction (
    compaction_id text PRIMARY KEY,
    cadence text NOT NULL CHECK (cadence IN ('weekly', 'monthly')),
    window_start timestamptz NOT NULL,
    window_end timestamptz NOT NULL CHECK (window_end > window_start),
    cycle_id text NOT NULL,
    run_id text NOT NULL UNIQUE,
    generation integer NOT NULL DEFAULT 0,
    status text NOT NULL CHECK (status IN ('pending', 'completed', 'empty', 'cancelled')),
    state jsonb NOT NULL,
    updated_at timestamptz NOT NULL,
    UNIQUE (cadence, window_start),
    CHECK (status <> 'completed' OR COALESCE(length(trim(state->>'body')) >= 80, false))
);
