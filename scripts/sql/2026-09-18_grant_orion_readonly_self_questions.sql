-- Let Orion's read-only sandbox role SELECT the self-question pool table.
-- Hub owns writes in v1; this grant is for future kickoff listing in the
-- sandbox and for parity with other self-inquiry outcome tables.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < scripts/sql/2026-09-18_grant_orion_readonly_self_questions.sql

BEGIN;

GRANT SELECT ON public.curiosity_self_questions TO orion_readonly;

COMMIT;

-- Verify:
--   SELECT has_table_privilege('orion_readonly', 'public.curiosity_self_questions', 'SELECT');
