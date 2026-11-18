-- Let Orion's read-only sandbox role read the hourly attention rollup that `dig attention` now reads
-- (orion/attention/frame_rollup.py; services/orion-sql-db/manual_migration_attention_hour_rollup_v1.sql).
-- Without it `python -P -m orion.dig attention` reads "needs GRANT SELECT ON substrate_attention_hour_rollup
-- TO orion_readonly". The 2026-11-17 grant on substrate_attention_frames stays (the live eval's E4 reads it).
--
-- Read-only grant; no write. Not a migration: GRANT is not on the post-pull auto-apply allowlist.
-- Apply AFTER the migration has created the table (the post-pull rebuild applies it before attention-runtime).
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-18_grant_orion_readonly_attention_hour_rollup.sql

BEGIN;

GRANT SELECT ON public.substrate_attention_hour_rollup TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.substrate_attention_hour_rollup FROM orion_readonly;
--
-- Verify (expect t), then `python -P -m orion.dig attention` prints rates, not a GRANT line:
--   SELECT has_table_privilege('orion_readonly', 'substrate_attention_hour_rollup', 'SELECT');
