-- Let Orion's read-only sandbox role read (1) walkway arrival expectations and (2) what each
-- dream replayed, so Orion can trace where a person's name enters a dream instead of guessing.
-- Why: on 2026-11-17 Orion found it had told Juniper camera-only people never reach its dreams;
-- dreams are fed walkway expectations that carry names. Tracing that needs these two tables.
-- Neither table holds pictures or face data: vision_percept_expectation carries the walkway
-- group's current label (Juniper's naming answer, or a placeholder like "person #4eb8e4") plus
-- window, status and scoring; dream_replay_item carries refs, rank, weight and a short reason.
-- Approved by Juniper 2026-11-18 ("yes") after the table description in the dev session.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < scripts/sql/2026-11-18_grant_orion_readonly_dream_replay_and_walkway_expectations.sql

BEGIN;

GRANT SELECT ON public.vision_percept_expectation TO orion_readonly;
GRANT SELECT ON public.dream_replay_item TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.vision_percept_expectation FROM orion_readonly;
--   REVOKE SELECT ON public.dream_replay_item FROM orion_readonly;
--
-- Verify:
--   SELECT has_table_privilege('orion_readonly', 'public.vision_percept_expectation', 'SELECT'),
--          has_table_privilege('orion_readonly', 'public.dream_replay_item', 'SELECT');
