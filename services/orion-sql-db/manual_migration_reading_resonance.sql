-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs, orion-substrate-runtime
-- The immutable request and checked result are the cross-service reading-resonance contract.
CREATE TABLE IF NOT EXISTS reading_resonance (
    seed_id text PRIMARY KEY REFERENCES world_pulse_read_seed(seed_id),
    run_id text NOT NULL UNIQUE,
    request_json jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    submitted_at timestamptz,
    result_json jsonb,
    completed_at timestamptz,
    outreach_note_reason text,
    CHECK ((result_json IS NULL) = (completed_at IS NULL))
);
CREATE INDEX IF NOT EXISTS reading_resonance_completed_idx ON reading_resonance(completed_at);
