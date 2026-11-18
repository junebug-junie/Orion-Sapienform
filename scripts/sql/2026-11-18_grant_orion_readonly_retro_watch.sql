-- Let Orion's read-only role read its own looks at the game screen (game company patch 2):
--
--   retro_watch_look        one row per look: the note, its outcome and verdict bucket, the frame hashes and
--                           when the frames went, the want and lease ids, the screen change.
--   retro_watch_floor       one row per game: the calibrated screen-change floor and the counts it came from.
--   retro_watch_preference  Juniper's switch and Orion's own watch choices (dig watch).
--   want_ledger             the market's decisions (every family). `dig watch status` prints the last refused
--                           look bid with its reason, lane and price, as the spec promises Orion ("the one who
--                           was refused can read why"). Live 2026-11-18: orion_readonly cannot read it.
--
-- `dig watch status` reads these as orion_readonly, and the spec's acceptance check says Orion's read-only
-- role can read retro_watch_look. Same pattern as scripts/sql/2026-11-17_grant_orion_readonly_retro.sql
-- (retro_sitting, retro_play_sessions; #3657), which this does not replace. No privacy boundary between
-- Juniper and Orion; no table here holds a name.
--
-- Read-only grant, not under services/orion-sql-db/: GRANT is not on the post-pull auto-apply allowlist.
-- Apply by hand (from the host), on Juniper's word, AFTER manual_migration_retro_watch_v2.sql is applied:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -f - < scripts/sql/2026-11-18_grant_orion_readonly_retro_watch.sql

BEGIN;

GRANT SELECT ON public.retro_watch_look TO orion_readonly;
GRANT SELECT ON public.retro_watch_floor TO orion_readonly;
GRANT SELECT ON public.retro_watch_preference TO orion_readonly;
GRANT SELECT ON public.want_ledger TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.retro_watch_look, public.retro_watch_floor, public.retro_watch_preference,
--     public.want_ledger FROM orion_readonly;
-- Verify (expect t, t, t, t, f):
--   SELECT has_table_privilege('orion_readonly', 'retro_watch_look', 'SELECT'),
--          has_table_privilege('orion_readonly', 'retro_watch_floor', 'SELECT'),
--          has_table_privilege('orion_readonly', 'retro_watch_preference', 'SELECT'),
--          has_table_privilege('orion_readonly', 'want_ledger', 'SELECT'),
--          has_table_privilege('orion_readonly', 'retro_watch_look', 'INSERT');
