-- Drive audit slim table v4: persist tick_attribution + tension_kinds.
-- Producer: orion-spark-concept-induction (memory.drives.audit.v1 on
-- orion:memory:drives:audit). Wire schema DriveAuditV1 already carries both
-- fields; sql-writer previously dropped them at the mapper-column filter because
-- they were not columns on DriveAuditSQL.
--
-- New consumer: Hub Drives Analytics
-- (docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md) — windowed
-- contributor history and live/window attribution cards. Bounded payload:
-- tick_attribution is the 6 fixed DRIVE_KEYS floats; tension_kinds is a short
-- list of kind strings per tick.
--
-- No backfill. Pre-migration rows remain NULL; the Hub UI must label windows
-- that mix attributed and null rows ("attribution not recorded before <ts>").
--
-- On boot, sql-writer applies the same ALTER TABLE ... ADD COLUMN IF NOT EXISTS
-- statements via app/main.py lifespan, so pre-existing deployments are upgraded
-- automatically on restart. This file is the standalone equivalent for manual
-- application against a running Postgres; it is a harmless no-op once applied.
--
-- Column ownership matches the convention in manual_migration_drive_audits_v1.sql
-- / services/orion-sql-writer/app/models/drive_audit.py: model declares the
-- columns; boot DDL + this migration keep live tables in sync.

ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tick_attribution JSONB;
ALTER TABLE drive_audits ADD COLUMN IF NOT EXISTS tension_kinds JSONB;
