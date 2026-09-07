-- World-pulse Stage 2 handoff persistence + Stage 2 claim columns
-- Apply: PGPASSWORD=postgres psql -h 127.0.0.1 -p 55432 -U postgres -d conjourney \
--   -f services/orion-sql-db/manual_migration_world_pulse_read_stage2_v1.sql

ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS handoff_json jsonb;

ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS handoff_at timestamptz;

ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS stage2_status text NOT NULL DEFAULT 'pending';

ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS stage2_claimed_at timestamptz;

ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS stage2_completed_at timestamptz;

ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS stage2_error text;

ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS stage2_trace_id text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'world_pulse_read_seed_stage2_status_check'
    ) THEN
        ALTER TABLE world_pulse_read_seed
            ADD CONSTRAINT world_pulse_read_seed_stage2_status_check
            CHECK (stage2_status IN ('pending', 'claimed', 'done', 'failed', 'skipped'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_world_pulse_read_seed_stage2_claim
    ON world_pulse_read_seed (stage2_status, priority, handoff_at);
