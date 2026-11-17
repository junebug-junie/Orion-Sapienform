-- Let Orion's read-only sandbox role read what `dig attention` (issue #3600, orion/dig/attention.py) needs.
-- Without it every `python -P -m orion.dig attention` reads "needs GRANT SELECT ON substrate_attention_frames TO
-- orion_readonly" (checked live 2026-11-17 as orion_readonly: permission denied). temporal_self_event is already
-- granted (2026-11-16_grant_orion_readonly_organ_roundup.sql).
--
--   substrate_attention_frames  world-first attention frames (~41k a day, pruned after ~72 h): which targets were
--                               eligible, which won, and the verdict for each one not attended. Orion's own
--                               attention; no Juniper text beyond what world:chat's reason already quotes.
--
-- Read-only grant; no write. Not a migration: GRANT is not on the post-pull auto-apply allowlist.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-17_grant_orion_readonly_attention_frames.sql

BEGIN;

GRANT SELECT ON public.substrate_attention_frames TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.substrate_attention_frames FROM orion_readonly;
--
-- Verify (expect t), then `python -P -m orion.dig attention` prints rates, not a GRANT line:
--   SELECT has_table_privilege('orion_readonly', 'substrate_attention_frames', 'SELECT');
