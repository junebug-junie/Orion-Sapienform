-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Bus traffic rollup, Patch 1 of docs/superpowers/specs/2026-11-11-causal-geometry-redesign.md.
--
-- bus_traffic_rollup_5m: one row per closed 5-minute bucket of bus traffic. Writer:
-- orion-sql-writer off bus orion:bus:traffic:rollup (BusTrafficRollupV1, kind
-- bus.traffic.rollup.v1; producer: orion-bus-mirror app/rate_rollup.py). Insert-only,
-- PK rollup_id = "<node>:<bucket_start iso>". channel_counts maps catalog-normalized
-- channel name -> message count; hops is a list of {from_organ, to_organ, count,
-- latency_mean_sec}. Reader: the Patch 2 replay eval.
-- sql-writer's create_all makes the same table at boot (index name matches SQLAlchemy's
-- index=True default); this file is the jsonb-typed operator contract. Additive only.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_bus_traffic_rollup_5m_v1.sql
-- Rollback: DROP TABLE bus_traffic_rollup_5m;  (loses the stored history)

CREATE TABLE IF NOT EXISTS bus_traffic_rollup_5m (
    rollup_id        text PRIMARY KEY,
    node             text NOT NULL,
    bucket_start     timestamptz NOT NULL,
    bucket_end       timestamptz NOT NULL,
    message_count    integer NOT NULL,
    channel_counts   jsonb NOT NULL,
    hops             jsonb NOT NULL,
    decode_failures  integer NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_bus_traffic_rollup_5m_bucket_start ON bus_traffic_rollup_5m (bucket_start);
