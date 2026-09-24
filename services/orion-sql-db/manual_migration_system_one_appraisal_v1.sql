-- Shadow System One appraisal frames produced by orion-substrate-runtime.
--
-- One writer only: services/orion-substrate-runtime/app/store.py::
-- save_system_one_appraisal(). Frames are compiled from already-live attention
-- artifacts and are deliberately behavior-inert until their live distribution
-- and calibration pass the repo metric-quality gate.
--
-- Apply:
--   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_system_one_appraisal_v1.sql

CREATE TABLE IF NOT EXISTS substrate_system_one_appraisal (
    frame_id text PRIMARY KEY,
    generated_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    provider text NOT NULL,
    model_id text NOT NULL,
    source_broadcast_projection_id text NOT NULL,
    source_field_attention_frame_id text,
    frame_json jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_substrate_system_one_appraisal_generated_at
    ON substrate_system_one_appraisal (generated_at DESC);

CREATE INDEX IF NOT EXISTS idx_substrate_system_one_appraisal_model
    ON substrate_system_one_appraisal (provider, model_id, generated_at DESC);
