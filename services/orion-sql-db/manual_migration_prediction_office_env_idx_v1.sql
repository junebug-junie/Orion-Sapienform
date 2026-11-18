-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction reads the office room sensor as series office.temp_c (source kind office_env,
-- services/orion-prediction/app/store.py):
--
--   SELECT ts, office_temp_c AS v FROM office_env_sample
--    WHERE node = %s AND ts >= %s AND ts < %s AND sensor_online AND NOT stale ... ORDER BY 1
--
-- every 30 s tick (issue, scoring) and over ~27 days at each selection. The table grows ~2,900
-- rows/day (one pan-1 SHT40 sample every 30 s; orion-sql-writer writes it). The existing
-- ix_office_env_sample_ts index covers ts alone; this one matches the (node, ts) access path so a
-- second room sensor on the same table does not turn every read into a filter over both.
--
-- Additive only. CONCURRENTLY: no write lock on the live table, and it cannot run inside a
-- transaction block, so this file has no BEGIN/COMMIT. An interrupted build leaves an INVALID
-- index that IF NOT EXISTS then skips. Check:
--   select indisvalid from pg_index where indexrelid = 'idx_office_env_sample_node_ts'::regclass;
-- Recover: DROP INDEX CONCURRENTLY IF EXISTS idx_office_env_sample_node_ts; then re-run this file.
--
-- Apply by hand (the post-pull rebuild applies it itself when it is NOT APPLIED):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_prediction_office_env_idx_v1.sql
-- Rollback: DROP INDEX CONCURRENTLY IF EXISTS idx_office_env_sample_node_ts;

create index concurrently if not exists idx_office_env_sample_node_ts
    on office_env_sample (node, ts);
