-- Substrate brain-frame log (Self tab brain-EKG realtime + playback backbone).
-- Append-per-frame, bounded by BRAIN_FRAME_RETENTION_HOURS prune in the producer.
CREATE TABLE IF NOT EXISTS substrate_brain_frame_log (
    frame_id TEXT PRIMARY KEY,
    tick_seq BIGINT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,
    phase TEXT NOT NULL,
    frame_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_brain_frame_generated
    ON substrate_brain_frame_log(generated_at DESC);
