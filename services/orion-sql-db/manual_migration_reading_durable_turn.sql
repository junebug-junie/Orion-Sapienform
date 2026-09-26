-- Apply before deploying Hub's durable reading adapter.
CREATE TABLE IF NOT EXISTS reading_durable_turn (
    seed_id text NOT NULL REFERENCES world_pulse_read_seed(seed_id),
    stage integer NOT NULL CHECK (stage IN (1, 2)),
    attempt integer NOT NULL,
    run_id text NOT NULL UNIQUE,
    request_json jsonb NOT NULL,
    consumed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (seed_id, stage, attempt)
);
