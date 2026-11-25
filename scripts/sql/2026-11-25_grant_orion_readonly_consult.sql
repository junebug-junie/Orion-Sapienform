-- Let Orion's read-only sandbox role read consults (Orion + Claude -> Juniper), so
-- `python -P -m orion.dig consult <id>` can open the one Juniper names in chat by the
-- "ref consult <id>" on its Hub card. Read-only; no other table changes.
-- Asked for by Juniper 2026-11-25 ("why don't we have an id ref number so I can talk to
-- orion in chat about what the hell this is about").
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-25_grant_orion_readonly_consult.sql

BEGIN;

GRANT SELECT ON public.orion_consult, public.orion_consult_turn TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.orion_consult, public.orion_consult_turn FROM orion_readonly;
--
-- Verify (both true):
--   SELECT has_table_privilege('orion_readonly', 'public.orion_consult', 'SELECT'),
--          has_table_privilege('orion_readonly', 'public.orion_consult_turn', 'SELECT');
