-- ROADMAP B1: raw physical units alongside the normalised pressures.
--
-- orion-sql-writer creates tables with `Base.metadata.create_all`, which creates missing
-- TABLES but never missing COLUMNS. An existing deployment therefore needs this ALTER; a
-- fresh one gets the column from the model and this file is a no-op.
--
-- Idempotent. Safe to re-run.
--
--   PGPASSWORD=... psql -h localhost -p 55432 -U postgres -d conjourney \
--     -f scripts/sql/2026-08-14_biometrics_measurements.sql

ALTER TABLE orion_biometrics_summary
    ADD COLUMN IF NOT EXISTS measurements JSONB;

COMMENT ON COLUMN orion_biometrics_summary.measurements IS
    'Raw physical quantities in their own units (chassis_watts, gpu_watts_total, '
    'fan_pct_max, temp_c_max, disk_bytes_per_sec, net_bytes_per_sec, cpu_cores, load_1m, '
    'load_15m). A key is ABSENT when unmeasured on that node -- never 0.0 -- so any fleet '
    'total summed from this column is honest about what it could not see. Read with '
    'value ? ''key'' rather than COALESCE(...,0).';
