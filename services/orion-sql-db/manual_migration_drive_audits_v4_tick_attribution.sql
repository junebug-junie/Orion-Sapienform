-- Drive audit v4: persist slim tick_attribution + tension_kinds for Hub Drives Analytics.
-- Wire schema DriveAuditV1 already carries these; sql-writer previously dropped them.
-- No backfill. Pre-migration rows remain NULL.
ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tick_attribution JSONB;
ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tension_kinds JSONB;
