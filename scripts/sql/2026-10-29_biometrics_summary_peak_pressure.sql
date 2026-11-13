-- Per-node binding constraint (peak_pressure, peak_pressure_channel) on orion_biometrics_summary.
--
-- BiometricsSummaryV1 has carried both fields since 2026-08-15, and orion_biometrics_cluster
-- already stores them. The per-node table had no columns for them, so _write_row (which filters
-- payload keys against the mapper's columns) dropped them on every row without an error.
--
-- YOU DO NOT NORMALLY NEED THIS FILE. orion-sql-writer applies the same statements at boot
-- (app/main.py), the same convention as 2026-08-14_biometrics_measurements.sql. This file is
-- for operators applying schema out-of-band. Idempotent, and safe when the table is absent.
--
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < scripts/sql/2026-10-29_biometrics_summary_peak_pressure.sql

ALTER TABLE IF EXISTS orion_biometrics_summary
    ADD COLUMN IF NOT EXISTS peak_pressure DOUBLE PRECISION;

ALTER TABLE IF EXISTS orion_biometrics_summary
    ADD COLUMN IF NOT EXISTS peak_pressure_channel TEXT;
