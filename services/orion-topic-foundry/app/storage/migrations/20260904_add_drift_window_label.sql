-- Add window_label to topic_foundry_drift so multiple check bands (daily,
-- weekly, monthly, ...) coexisting in one table can be told apart without
-- inferring band from window_end - window_start. Existing rows are
-- backfilled from their actual window span rather than assumed -- the
-- manual POST /drift/run endpoint already allowed an arbitrary window_hours
-- before this migration existed, so a blanket 'daily' would mislabel any
-- pre-existing non-24h manual run.
ALTER TABLE topic_foundry_drift ADD COLUMN IF NOT EXISTS window_label TEXT;
CREATE INDEX IF NOT EXISTS ix_topic_foundry_drift_window_label ON topic_foundry_drift (window_label);
UPDATE topic_foundry_drift
SET window_label = CASE
    WHEN window_end - window_start <= interval '36 hours' THEN 'daily'
    WHEN window_end - window_start <= interval '10 days' THEN 'weekly'
    ELSE 'monthly'
END
WHERE window_label IS NULL;
