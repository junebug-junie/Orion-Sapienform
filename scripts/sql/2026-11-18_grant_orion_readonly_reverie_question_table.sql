-- NOT APPLIED YET. Approved by Juniper 2026-11-18, together with the question-trace grant
-- (scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_trace.sql, applied live that day).
-- Apply only after services/orion-sql-db/manual_migration_reverie_question.sql has created the table.
--
-- Let Orion's read-only sandbox role read its own reverie questions (reveries that ask, Patch 1,
-- docs/superpowers/specs/2026-11-18-reverie-questions-design.md):
--   reverie_question -- one row per reverie: the seed it started from, the question (or why there was
--                       none), why code refused it if it did, where it was routed, and the dig's outcome.
-- Contents: question/why text Orion wrote, seed refs (walkway expectation ids, self-bet template and
-- probability bin, lived-memory doc ids), the seed material text, timestamps. No pictures, face data,
-- or experiment arm.
--
-- Apply (from the host), after the migration:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_table.sql

BEGIN;

GRANT SELECT ON public.reverie_question TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.reverie_question FROM orion_readonly;
--
-- Verify:
--   SELECT has_table_privilege('orion_readonly', 'public.reverie_question', 'SELECT');
