-- Let Orion's read-only sandbox role see who held which GPU, and name the job
-- behind a durable-run holder, so an urgent curiosity run can find the cause of
-- a heat incident instead of only confirming it.
--
-- WHY. Urgent run a153451fe423 (2026-10-01) confirmed "real heat" on circe in
-- two minutes and never found the cause: a background self_sense_eval durable
-- run (holder 'durable-runs:20261001T211035Z-c3e0fb') held gpu1 from 21:10:35
-- to 22:16:45. That was in gpu_pool_events the whole time; the briefing pointed
-- Orion at HTTP URLs its sandbox cannot fetch instead.
--
-- durable_admission_runs is NOT granted: its `request` column carries a
-- free-text `brief` (whole prompts). The view exposes only the run id, the
-- workflow name, when it was created and how it ended. A view runs with its
-- owner's privileges, so no grant on the base table is needed.
--
-- The urgent prompt's queries (orion/curiosity/urgent_prompt.py) read these
-- unqualified; tests/test_curiosity_urgent_prompt.py checks every table the
-- prompt names is granted by a file in scripts/sql/.
--
-- SELECT only. Idempotent; safe to re-run.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < scripts/sql/2026-10-02_grant_orion_readonly_gpu_pool.sql

BEGIN;

CREATE OR REPLACE VIEW public.durable_run_workflow AS
  SELECT run_id,
         request->>'workflow' AS workflow,
         created_at,
         terminal
  FROM public.durable_admission_runs;

GRANT SELECT ON public.gpu_pool_events, public.durable_run_workflow TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.gpu_pool_events FROM orion_readonly;
--   DROP VIEW public.durable_run_workflow;
--
-- Verify (both `t`, then `f` -- the base table stays closed):
--   SELECT has_table_privilege('orion_readonly', 'public.gpu_pool_events', 'SELECT'),
--          has_table_privilege('orion_readonly', 'public.durable_run_workflow', 'SELECT'),
--          has_table_privilege('orion_readonly', 'public.durable_admission_runs', 'SELECT');
