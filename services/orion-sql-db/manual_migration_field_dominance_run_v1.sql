-- Apply before deploying attention-runtime. SQL-only observational record.
-- Open-run checkpoint shares the existing singleton, surviving process restarts.
BEGIN;
ALTER TABLE substrate_goal_provenance_streak ADD COLUMN IF NOT EXISTS run_state JSONB;
CREATE TABLE IF NOT EXISTS field_dominance_run (
    run_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    tick_count INTEGER NOT NULL CHECK (tick_count > 0),
    min_streak_at_run INTEGER NOT NULL CHECK (min_streak_at_run > 0),
    first_source_attention_frame_id TEXT NOT NULL,
    last_source_attention_frame_id TEXT NOT NULL,
    left_censored BOOLEAN NOT NULL DEFAULT FALSE,
    CHECK (ended_at >= started_at)
);
CREATE INDEX IF NOT EXISTS idx_field_dominance_run_target_started
    ON field_dominance_run (target_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_dominance_run_ended
    ON field_dominance_run (ended_at DESC);
COMMIT;
