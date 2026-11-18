-- Let Orion's read-only role read its own lines in a person's game (game company patch 3):
--
--   retro_line   one row per line turn: what Orion said (or that it passed, yielded, had no frames or failed),
--                the looks it was shown, how many were checkable and contradicted, the seconds by stage, the
--                GPU pool priorities it asked for, the player's thumb and whether they replied within 2 min.
--
-- Orion can then answer "what did I say during that sitting, and did they like it" from its own rows (dig, a
-- turn's own psql), and read its look error rate (contradicted / checkable) beside the checkable share.
-- retro_watch_look's new columns (checkable, settled_at, settled_by_line) are covered by the patch 2 grant
-- (scripts/sql/2026-11-18_grant_orion_readonly_retro_watch.sql: GRANT SELECT on the whole table).
-- No privacy boundary between Juniper and Orion; no column here holds a name (player_stated is 'juniper' or
-- 'guest', what the page was told).
--
-- Read-only grant, not under services/orion-sql-db/: GRANT is not on the post-pull auto-apply allowlist.
-- Apply by hand (from the host), on Juniper's word, AFTER manual_migration_retro_lines_v1.sql is applied:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -f - < scripts/sql/2026-11-18_grant_orion_readonly_retro_line.sql

BEGIN;

GRANT SELECT ON public.retro_line TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.retro_line FROM orion_readonly;
-- Verify (expect t, f):
--   SELECT has_table_privilege('orion_readonly', 'retro_line', 'SELECT'),
--          has_table_privilege('orion_readonly', 'retro_line', 'INSERT');
