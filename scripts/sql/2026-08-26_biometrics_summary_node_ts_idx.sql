-- Hub ambient history: node + time-range queries on orion_biometrics_summary.
--
-- orion-sql-writer applies the same statement at boot (app/main.py), which is the
-- existing convention for this service. This file exists for operators applying
-- schema out-of-band.
--
-- Idempotent, and safe against a database where the table does not exist yet.
--
--   PGPASSWORD=... psql -h localhost -p 55432 -U postgres -d conjourney \
--     -f scripts/sql/2026-08-26_biometrics_summary_node_ts_idx.sql

CREATE INDEX IF NOT EXISTS orion_biometrics_summary_node_ts_idx
    ON orion_biometrics_summary (node, timestamp);
