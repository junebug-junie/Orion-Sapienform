-- Let Orion's read-only role read the supersede trail (issue #3587), so Orion can verify the
-- supersede pass and write-time supersedes itself:
--
--   episode_memory_event   op 'superseded' / 'supersede_undone' rows: old memory id, the
--                          correction's id (evidence->>'superseded_by'), actor, reason, time.
--
-- Needs PR #3593's grant (scripts/sql/2026-11-17_grant_orion_readonly_mesh_edge.sql: SELECT on
-- episode_memory) for the join to statuses. The event table also holds the distiller's
-- created/rejected_invalid rows (their evidence can quote a rejected candidate). Same privacy
-- boundary as #3593: orion_readonly is the curiosity and FCC sandbox role, and there is no privacy
-- boundary between Juniper and Orion; named here on purpose.
--
-- Read-only grant, not under services/orion-sql-db/: GRANT is not on the post-pull auto-apply
-- allowlist. Apply by hand (from the host):
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON public.episode_memory_event TO orion_readonly;"

BEGIN;

GRANT SELECT ON public.episode_memory_event TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.episode_memory_event FROM orion_readonly;
-- Verify (expect t, f):
--   SELECT has_table_privilege('orion_readonly', 'episode_memory_event', 'SELECT'),
--          has_table_privilege('orion_readonly', 'episode_memory_event', 'INSERT');
