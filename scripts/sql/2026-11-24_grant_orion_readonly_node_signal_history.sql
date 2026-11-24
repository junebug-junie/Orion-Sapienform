-- Let Orion's read-only sandbox role read the signal history for the five non-PE nodes
-- (alarms: cabinet, strain, street; camera scene change: perception, perception_walkway).
-- Table: services/orion-sql-db/manual_migration_node_signal_history_v1.sql (PR #3781 Patch 1, #3800).
-- Later patches point Orion's node-history dig tool at this table, so orion_readonly must read it.
--
-- Today the postgres role's default privileges already give orion_readonly SELECT on new tables
-- (checked live 2026-11-24: has_table_privilege = t). This file makes the grant explicit, so it
-- does not depend on that default surviving a rebuild of the database.
--
-- Read-only grant; no write. Not a migration: GRANT is not on the post-pull auto-apply allowlist.
-- Apply AFTER the migration has created the table. Idempotent: re-running is safe.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-24_grant_orion_readonly_node_signal_history.sql

BEGIN;

GRANT SELECT ON public.substrate_node_signal_history TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.substrate_node_signal_history FROM orion_readonly;
--
-- Verify (expect t):
--   SELECT has_table_privilege('orion_readonly', 'substrate_node_signal_history', 'SELECT');
