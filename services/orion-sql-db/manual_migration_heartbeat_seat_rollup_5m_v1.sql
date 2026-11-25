-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Heartbeat seat rollup, issue #3886.
--
-- heartbeat_seat_rollup_5m: one row per heartbeat seat per closed 5-minute bucket. Writer:
-- orion-sql-writer off bus orion:heartbeat:seat:rollup (HeartbeatSeatRollupV1, kind
-- heartbeat.seat.rollup.v1; producer: orion-heartbeat app/seat_rollup.py). Insert-only,
-- PK rollup_id = "<node>:<seat>:<bucket_start iso>". level_sum = summed route_atom level of the
-- seat's atoms whose emitted_at falls in the bucket (the value scripts/demon_network_replay.py
-- recomputes from grammar_events); atom_count = those atoms; skipped_count = the seat's atoms the
-- input rule skipped. partial = the heartbeat was not listening for part of the bucket; readers
-- treat it as missing. A bucket with no rows is unknown, never zero.
-- Reader: scripts/demon_network_replay.py. grammar_events keeps ~3 days; this table is kept.
-- sql-writer's create_all makes the same table at boot (index name matches SQLAlchemy's
-- index=True default); this file is the operator contract. Additive only.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v1.sql
-- Rollback: DROP TABLE heartbeat_seat_rollup_5m;  (loses the stored history)

CREATE TABLE IF NOT EXISTS heartbeat_seat_rollup_5m (
    rollup_id      text PRIMARY KEY,
    node           text NOT NULL,
    seat           text NOT NULL,
    site_index     integer NOT NULL,
    bucket_start   timestamptz NOT NULL,
    bucket_end     timestamptz NOT NULL,
    level_sum      double precision NOT NULL,
    atom_count     integer NOT NULL,
    skipped_count  integer NOT NULL,
    partial        boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS ix_heartbeat_seat_rollup_5m_bucket_start ON heartbeat_seat_rollup_5m (bucket_start);
