-- Additive; retain on rollback. Same database and lock as durable capacity.
CREATE TABLE IF NOT EXISTS durable_elastic_slot (
    slot text PRIMARY KEY CHECK (slot = 'circe-gpu2'),
    backend_key text NOT NULL,
    generation bigint NOT NULL DEFAULT 0,
    operation_id text,
    run_id text,
    desired_target text NOT NULL DEFAULT 'diffusion' CHECK (desired_target IN ('diffusion','agent-burst')),
    state text NOT NULL DEFAULT 'idle' CHECK (state IN ('idle','requested','ready','failed','restoring')),
    admission_observed boolean NOT NULL DEFAULT false,
    admissions_open boolean NOT NULL DEFAULT false,
    requested_at timestamptz,
    resident_at timestamptz,
    idle_since timestamptz,
    last_restored_at timestamptz,
    detail jsonb NOT NULL DEFAULT '{}'
);
