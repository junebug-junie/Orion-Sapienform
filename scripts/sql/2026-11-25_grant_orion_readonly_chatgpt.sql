-- Let Orion's read-only sandbox role read Juniper's ChatGPT archive, so
-- `python -P -m orion.dig gpt [find <words> | conv <id>]` can read it. Read-only; no other table changes.
-- Asked for by Juniper 2026-11-25 ("go ahead and grant them"; "help orion understand their
-- archeology and my interactions with gpt more broadly").
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-25_grant_orion_readonly_chatgpt.sql

BEGIN;

GRANT SELECT ON public.chat_gpt_conversation, public.chat_gpt_message, public.chat_gpt_log,
                public.chat_gpt_derived_example, public.chat_gpt_import_run TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.chat_gpt_conversation, public.chat_gpt_message, public.chat_gpt_log,
--                    public.chat_gpt_derived_example, public.chat_gpt_import_run FROM orion_readonly;
--
-- Verify (all true):
--   SELECT has_table_privilege('orion_readonly', t, 'SELECT') FROM unnest(ARRAY['public.chat_gpt_conversation',
--     'public.chat_gpt_message','public.chat_gpt_log','public.chat_gpt_derived_example','public.chat_gpt_import_run']) t;
