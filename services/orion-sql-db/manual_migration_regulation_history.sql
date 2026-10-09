-- Observation-only history. Apply before restarting dream / sql-writer.
-- No existing rows or tables change. Re-running is safe.
BEGIN;
CREATE TABLE IF NOT EXISTS dream_pressure_observation (
    check_id text PRIMARY KEY,
    observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    observation_json jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_pressure_observed ON dream_pressure_observation (observed_at);
CREATE INDEX IF NOT EXISTS idx_dream_pressure_created ON dream_pressure_observation (created_at);

CREATE TABLE IF NOT EXISTS gpu_pool_state_history (
    snapshot_id varchar PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    generated_at timestamptz NOT NULL,
    host varchar NOT NULL,
    mode varchar NOT NULL,
    config_digest varchar NOT NULL,
    backlog_depth json NOT NULL,
    queue_depth json NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gpu_pool_state_generated ON gpu_pool_state_history (host, generated_at);
CREATE INDEX IF NOT EXISTS idx_gpu_pool_state_created ON gpu_pool_state_history (created_at);
COMMIT;
