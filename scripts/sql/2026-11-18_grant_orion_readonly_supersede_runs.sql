-- Let Orion's read-only role read the supersede audit trail and eval verdicts (issues #3659, #3660):
--
--   memory_supersede_run       one row per supersede run: ids, verdicts, votes, which candidate legs
--                              fired, and the model's one-sentence reasons (these may paraphrase a
--                              memory; the statements themselves are never stored here).
--   memory_supersede_eval_run  one row per precision eval: counts, precision, recall, passed.
--
-- Same privacy boundary as scripts/sql/2026-11-17_grant_orion_readonly_supersede_events.sql.
-- GRANT is not on the post-pull auto-apply allowlist. Apply by hand after
-- services/orion-sql-db/manual_migration_episode_memory_temporal_v1.sql:
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON public.memory_supersede_run, public.memory_supersede_eval_run TO orion_readonly;"

BEGIN;

GRANT SELECT ON public.memory_supersede_run TO orion_readonly;
GRANT SELECT ON public.memory_supersede_eval_run TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.memory_supersede_run, public.memory_supersede_eval_run FROM orion_readonly;
-- Verify (expect t, t):
--   SELECT has_table_privilege('orion_readonly', 'memory_supersede_run', 'SELECT'),
--          has_table_privilege('orion_readonly', 'memory_supersede_eval_run', 'SELECT');
