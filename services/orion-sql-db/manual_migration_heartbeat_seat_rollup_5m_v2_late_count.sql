-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-heartbeat
-- Heartbeat seat rollup late_count, issue #3901.
--
-- late_count: of atom_count, the atoms that arrived after bucket_end, inside the heartbeat's flush
-- grace (HeartbeatSeatRollupV1.late_count; producer orion-heartbeat app/seat_rollup.py). NULL for
-- rows written before this column, by a heartbeat that did not measure it. Atoms that arrive after
-- the grace are not in the row; the heartbeat counts them on /health (late_atoms_dropped).
-- Nullable, no default, no backfill. Additive only. sql-writer's create_all does not add a column
-- to an existing table, so apply this before the sql-writer that writes late_count starts.
-- orion-heartbeat is listed too: HeartbeatSeatRollupV1 has extra="forbid", so an old sql-writer
-- rejects rows that carry late_count. Deploy order: this file, orion-sql-writer, orion-heartbeat.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v2_late_count.sql
-- Rollback: ALTER TABLE heartbeat_seat_rollup_5m DROP COLUMN late_count;  (loses the stored counts)

ALTER TABLE heartbeat_seat_rollup_5m ADD COLUMN IF NOT EXISTS late_count integer;
