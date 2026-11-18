-- ORION-MIGRATION-REQUIRED-BY: none drops a dead table; no service reads or writes substrate_prediction_raw_reading since 2026-11-03
-- DESTRUCTIVE
--
-- Retire substrate_prediction_raw_reading (issue #3505). Applied BY HAND only, after Juniper's
-- approval; the post-pull applier never runs a DESTRUCTIVE file.
--
-- Why it is dead: orion-substrate-runtime stopped writing it on 2026-11-03 (rung 6 tuner retirement,
-- docs/superpowers/pr-reports/2026-11-03-rung6-retire-tuner-pr.md) and the Hub tuner that read it is
-- deleted. Re-checked 2026-11-18: no code reference outside manual_migration_prediction_raw_reading_v1.sql
-- and docs; last rows 2026-11-03 09:44Z; no view, foreign key, trigger or function depends on it.
--
-- Snapshot (pg_dump -Fc, taken 2026-11-18, 16,100 rows: execution 14,209 / chat 1,879 / street_handoff 12,
-- sha256 18146ca75d6ddd92be0f698dd0cc4ad630d8d302880c1f46358d2f99c11754c3):
--   /mnt/storage-warm/orion-backups/substrate_prediction_raw_reading_2026-11-18.dump   (durable)
--   /tmp/substrate_prediction_raw_reading_retire/snapshot.dump                        (same file; /tmp clears on reboot)
--
-- Check first that nothing has written since (must print 2026-11-03 09:44:50...):
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select max(observed_at) from substrate_prediction_raw_reading"
-- Apply:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_retire_prediction_raw_reading_drop.sql
-- Restore (recreates table, rows, primary key and index):
--   docker exec -i orion-athena-sql-db pg_restore -U postgres -d conjourney < /mnt/storage-warm/orion-backups/substrate_prediction_raw_reading_2026-11-18.dump
--
-- manual_migration_prediction_raw_reading_v1.sql stays: the repo keeps a retired table's create file and
-- lets the later drop file supersede it (same as the feedback/consolidation rung retirement).

BEGIN;
SET LOCAL lock_timeout = '10s';
DROP TABLE IF EXISTS substrate_prediction_raw_reading;
COMMIT;
