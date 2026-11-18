-- Let Orion's read-only sandbox role read what `dig organs <date>` (the daily organ roundup,
-- orion/dig/organs.py; integration patch 4) needs. Without it every roundup query reads
-- "needs GRANT SELECT ON <table> TO orion_readonly" and Orion's own roundup is all UNREAD lines
-- (checked live 2026-11-16 as orion_readonly). The day letter's copy of the roundup is unaffected:
-- Hub reads it with its own connection.
--
--   temporal_self_event  Orion's experiences (labels, payloads; juniper_chat items included, as in the
--                        now every organ already reads -- no privacy boundary between Juniper and Orion)
--   experience_shown     which experiences an organ was shown
--   experience_use       which it cited
--   grammar_acts         the act ledger (acts per organ; Prime's "woke:" notes)
--   notify_requests      the notification counts and critical titles of the retired Daily Digest
--                        (orion_readonly already reads chat_history_log, the same chat text)
--
-- Read-only grants; no write. Not a migration: GRANT is not on the post-pull auto-apply allowlist.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < scripts/sql/2026-11-16_grant_orion_readonly_organ_roundup.sql

BEGIN;

-- Superseded for temporal_self_event by 2026-11-18_blind_experiment_arms_orion_readonly.sql (#3458): the role reads
-- orion_scope.temporal_self_event, a view without blinded experiment arms. Re-running this line reopens the arm.
GRANT SELECT ON public.temporal_self_event TO orion_readonly;
GRANT SELECT ON public.experience_shown TO orion_readonly;
GRANT SELECT ON public.experience_use TO orion_readonly;
GRANT SELECT ON public.grammar_acts TO orion_readonly;
GRANT SELECT ON public.notify_requests TO orion_readonly;
-- act_downstream_use: the integration metric's "before" baseline (Orion's question 7 on #3528: it could not
-- check its own integration priors).
GRANT SELECT ON public.act_downstream_use TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.temporal_self_event, public.experience_shown, public.experience_use,
--     public.grammar_acts, public.notify_requests FROM orion_readonly;
--
-- Verify (expect t | t | t | t | t), then `python -P -m orion.dig organs <yesterday>` prints no UNREAD line:
--   SELECT has_table_privilege('orion_readonly', 'temporal_self_event', 'SELECT'),
--          has_table_privilege('orion_readonly', 'experience_shown', 'SELECT'),
--          has_table_privilege('orion_readonly', 'experience_use', 'SELECT'),
--          has_table_privilege('orion_readonly', 'grammar_acts', 'SELECT'),
--          has_table_privilege('orion_readonly', 'notify_requests', 'SELECT'),
--          has_table_privilege('orion_readonly', 'act_downstream_use', 'SELECT');
