-- Let Orion's read-only role read its own play record (game company patch 1, PR #3627):
--
--   retro_sitting          one row per sitting: game, start/end, mask changes, who said they played.
--   retro_play_sessions    one row per finished play: Orion's own leisure play and human_play rows.
--
-- Orion asked for this in review of PR #3632 (thread 4c9a86f14377, point 9): as orion_readonly both
-- tables answer "permission denied" today (live 2026-11-17: information_schema shows SELECT for
-- postgres only). No privacy boundary between Juniper and Orion; player_stated holds a first name
-- or "guest", nothing else.
--
-- Read-only grant, not under services/orion-sql-db/: GRANT is not on the post-pull auto-apply
-- allowlist. Apply by hand (from the host), on Juniper's word:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -f - < scripts/sql/2026-11-17_grant_orion_readonly_retro.sql

BEGIN;

GRANT SELECT ON public.retro_sitting TO orion_readonly;
GRANT SELECT ON public.retro_play_sessions TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.retro_sitting, public.retro_play_sessions FROM orion_readonly;
-- Verify (expect t, t, f):
--   SELECT has_table_privilege('orion_readonly', 'retro_sitting', 'SELECT'),
--          has_table_privilege('orion_readonly', 'retro_play_sessions', 'SELECT'),
--          has_table_privilege('orion_readonly', 'retro_sitting', 'INSERT');
