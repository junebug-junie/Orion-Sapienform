-- Let Orion's read-only sandbox role SELECT its own outcome tables, for the
-- curiosity self-inquiry line (orion/curiosity/self_inquiry.py).
--
-- WHY. The investigation line reads four tables (see
-- scope_orion_readonly_off_aitown.sql). "What am I, and what am I made of?"
-- needs the records of what Orion has actually done: dreams, motor turns,
-- reverie chains, attention frames, stance beliefs, self-knowledge items and
-- previous self-definitions. Without these grants a self-inquiry turn spends
-- its budget on "permission denied" and journals that as a finding.
--
-- Hub CHECKS these grants before every self-inquiry run
-- (SELF_INQUIRY_GRANTS_SQL, gate reason `pg_grants_missing`), so applying
-- this script is what turns the line on; the flag alone does not.
--
-- SELECT only. The role still cannot write anything. Idempotent; safe to
-- re-run.
--
-- 2026-09-10: added self_sense_eval_log (PR #2171) -- Orion's own scored
-- chat answers, including a real caught instance of the exact chatbot/
-- customer-support framing this arc exists to fix (run
-- 20260910T022934Z-e863f0, question "what can't you do"). Without this
-- grant a self-inquiry run cannot see how its own definition is actually
-- landing in chat. `orion_scope` search_path from the AI Town scoping script is
-- unaffected: none of these tables are shadowed there.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < scripts/sql/2026-09-08_grant_orion_readonly_self_inquiry.sql
--
-- TO UNDO, in full:
--   REVOKE SELECT ON public.dreams, public.harness_turn_trace,
--     public.substrate_reverie_chain, public.substrate_reverie_thought,
--     public.reverie_visual_chain,
--     public.substrate_attention_schema, public.chat_stance_belief_log,
--     public.self_knowledge_items, public.self_concept_history,
--     public.substrate_endogenous_curiosity_candidates FROM orion_readonly;

BEGIN;

GRANT SELECT ON
  public.dreams,
  public.harness_turn_trace,
  public.substrate_reverie_chain,
  public.substrate_reverie_thought,
  public.reverie_visual_chain,
  public.substrate_attention_schema,
  public.chat_stance_belief_log,
  public.self_knowledge_items,
  public.self_concept_history,
  public.substrate_endogenous_curiosity_candidates,
  public.self_sense_eval_log
TO orion_readonly;

COMMIT;

-- Verify (each row should be `t`):
--   SELECT t, has_table_privilege('orion_readonly', 'public.' || t, 'SELECT')
--   FROM unnest(ARRAY['dreams','harness_turn_trace','substrate_reverie_chain',
--     'substrate_reverie_thought','reverie_visual_chain','substrate_attention_schema','chat_stance_belief_log',
--     'self_knowledge_items','self_concept_history',
--     'substrate_endogenous_curiosity_candidates','self_sense_eval_log']) AS t;
