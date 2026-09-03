-- Latest-induction-per-node reads on orion_biometrics_induction.
--
-- Backs orion/substrate/metacog_trend_signals.py::latest_biometrics_induction_by_node:
--   SELECT DISTINCT ON (node) ... ORDER BY node, timestamp DESC
--
-- Without this index that query parallel-seq-scans the whole table (247MB /
-- 187k rows) and external-merge-sorts it to disk to return one row per node.
-- Measured live 2026-09-03 via pg_stat_statements: 418ms mean over 36,393
-- calls, 11.5 TB of temp spill -- 92% of all temp I/O on the instance -- once
-- the Hub's Biometrics card started polling /api/biometrics/preview/induction
-- every 10s per node.
--
-- DESC is load-bearing: it is the direction the DISTINCT ON reads.
--
-- orion-sql-writer applies the same statement at boot (app/main.py), which is
-- the existing convention for this service. This file exists for operators
-- applying schema out-of-band, and for applying it CONCURRENTLY on a live
-- instance without taking a write lock (see the commented form below).
--
-- Idempotent, and safe against a database where the table does not exist yet.
--
--   PGPASSWORD=... psql -h localhost -p 55432 -U postgres -d conjourney \
--     -f scripts/sql/2026-09-03_biometrics_induction_node_ts_idx.sql

CREATE INDEX IF NOT EXISTS orion_biometrics_induction_node_ts_idx
    ON orion_biometrics_induction (node, timestamp DESC);

-- Live-instance form -- cannot run inside a transaction block, so it is not
-- part of the boot DDL above:
--
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS orion_biometrics_induction_node_ts_idx
--       ON orion_biometrics_induction (node, timestamp DESC);
