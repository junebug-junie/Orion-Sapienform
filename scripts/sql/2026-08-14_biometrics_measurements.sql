-- ROADMAP B1: raw physical units alongside the normalised pressures.
--
-- orion-sql-writer creates tables with `Base.metadata.create_all`, which creates missing
-- TABLES but never missing COLUMNS.
--
-- YOU DO NOT NORMALLY NEED THIS FILE. orion-sql-writer applies the same statement at boot
-- (app/main.py), which is the existing convention for this hazard and removes the
-- deploy-ordering trap: the writer declaring the column while the DB lacks it turns every
-- biometrics-summary INSERT into UndefinedColumn -> ProgrammingError, stopping ALL of that
-- table's persistence -- not just the new field -- while the bus publish and the Hub panel
-- keep looking healthy. This file exists for operators applying schema out-of-band.
--
-- Idempotent, and safe against a database where the table does not exist yet (an earlier
-- draft omitted `IF EXISTS`, so the copy-paste invocation below failed with
-- `relation does not exist` on a fresh DB -- the exact case its own header claimed was a
-- no-op).
--
--   PGPASSWORD=... psql -h localhost -p 55432 -U postgres -d conjourney \
--     -f scripts/sql/2026-08-14_biometrics_measurements.sql

ALTER TABLE IF EXISTS orion_biometrics_summary
    ADD COLUMN IF NOT EXISTS measurements JSONB;

COMMENT ON COLUMN orion_biometrics_summary.measurements IS
    'Raw physical quantities in their own units (chassis_watts, gpu_watts_total, '
    'gpu_count, fan_pct_max, temp_c_max, cpu_cores, load_1m, load_15m). A key is ABSENT '
    'when unmeasured on that node -- never 0.0 -- so any fleet total summed from this column '
    'is honest about what it could not see. Read with value ? ''key'' rather than '
    'COALESCE(...,0). NULL means the producer predates this column and says nothing about '
    'the node; ''{}'' means a current producer measured nothing. CONTAINMENT: chassis_watts '
    'is measured at the PSU and ALREADY INCLUDES gpu_watts_total -- never sum the two.';
