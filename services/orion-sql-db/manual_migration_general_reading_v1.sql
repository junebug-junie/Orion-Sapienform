-- Additive general reading ingress; retains the existing queue and workers.
-- Apply after both world_pulse_read migrations, before deploying Hub.
ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS request_id uuid,
    ADD COLUMN IF NOT EXISTS request_json jsonb,
    ADD COLUMN IF NOT EXISTS root_request_id uuid,
    ADD COLUMN IF NOT EXISTS duplicate_of text,
    ADD COLUMN IF NOT EXISTS stage2_result_json jsonb,
    ADD COLUMN IF NOT EXISTS landing_at timestamptz;
ALTER TABLE world_pulse_read_seed DROP CONSTRAINT IF EXISTS world_pulse_read_seed_kind_check;
ALTER TABLE world_pulse_read_seed ADD CONSTRAINT world_pulse_read_seed_kind_check
    CHECK (kind IN ('finding', 'digest_item', 'reading'));
CREATE UNIQUE INDEX IF NOT EXISTS idx_reading_request_id ON world_pulse_read_seed(request_id);
CREATE INDEX IF NOT EXISTS idx_reading_active_url ON world_pulse_read_seed(url)
    WHERE status IN ('pending', 'claimed', 'done');
CREATE INDEX IF NOT EXISTS idx_reading_root ON world_pulse_read_seed(root_request_id);
