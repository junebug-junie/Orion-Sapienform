-- NOT APPLIED YET. Needs Juniper's yes (same shape as
-- scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_table.sql).
-- Apply only after services/orion-sql-db/manual_migration_reverie_question_threads.sql has created the table.
--
-- Let Orion's read-only sandbox role read its own reverie threads (reveries that ask, Patch 3,
-- docs/superpowers/specs/2026-11-18-reverie-questions-design.md, "Threads (idea 9) and read-back"):
--   reverie_thread -- one row per thread of reverie questions: its status (open, concluded, exhausted,
--                     lapsed), why, the last question on it, and when it last moved.
-- Contents: thread ids, seed kind, the thread key (a person's routine id, a bet template and bin, or two
-- lived-memory doc ids), status words, timestamps. No question text (that is in reverie_question, already
-- granted), no pictures, no face data.
--
-- Apply (from the host), after the migration:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < scripts/sql/2026-11-25_grant_orion_readonly_reverie_thread_table.sql

BEGIN;

GRANT SELECT ON public.reverie_thread TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.reverie_thread FROM orion_readonly;
--
-- Verify:
--   SELECT has_table_privilege('orion_readonly', 'public.reverie_thread', 'SELECT');
