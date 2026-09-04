-- Add window_label to topic_foundry_drift so multiple check bands (daily,
-- weekly, monthly, ...) coexisting in one table can be told apart without
-- inferring band from window_end - window_start. Existing rows (all
-- computed with the pre-band-system single 24h default) are backfilled to
-- 'daily'.
ALTER TABLE topic_foundry_drift ADD COLUMN IF NOT EXISTS window_label TEXT;
CREATE INDEX IF NOT EXISTS ix_topic_foundry_drift_window_label ON topic_foundry_drift (window_label);
UPDATE topic_foundry_drift SET window_label = 'daily' WHERE window_label IS NULL;
