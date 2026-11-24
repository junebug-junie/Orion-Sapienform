-- APPLIED live 2026-11-18 after Juniper approved (design PR #3736, missing question 5). Checked read-only on
-- 2026-11-18: has_table_privilege('orion_readonly', ...) is true for all three tables.
--
-- Let Orion's read-only sandbox role read the trace of its own reverie questions:
--   curiosity_focus_request    -- was the question queued, offered, taken, refused; the run's disposition
--   curiosity_finding_register -- did the dig find anything (operational / philosophical / no_finding, novelty)
--   prediction_turn_self_bet   -- Orion's own self-bets: the lost-bet seed's material
-- Why: Orion asked on PR #3736 (2026-11-18): without these it has to ask a session whether its own
-- question was ever dug. The `reverie_question` table's grant is a separate, not yet applied file:
-- scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_table.sql.
-- Contents: question/why text Orion or Juniper wrote, statuses, timestamps, refs, judge verdicts,
-- bet probabilities and outcomes. No pictures, face data, or stored experiment arm.
-- Run ids (#3685): curiosity_focus_request.offered_run_id and curiosity_finding_register.run_id are
-- curiosity run ids, from which the three-arm test's arm is re-derivable. orion_readonly already reads
-- those run ids through orion_scope.temporal_self_event (curiosity_run source_ref: 94 in the 7 days to
-- 2026-11-18, checked with SET ROLE orion_readonly), so this grant does not widen that gap.
--
-- Apply (from the host; already live, kept for the record):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_trace.sql

BEGIN;

GRANT SELECT ON public.curiosity_focus_request TO orion_readonly;
GRANT SELECT ON public.curiosity_finding_register TO orion_readonly;
GRANT SELECT ON public.prediction_turn_self_bet TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.curiosity_focus_request FROM orion_readonly;
--   REVOKE SELECT ON public.curiosity_finding_register FROM orion_readonly;
--   REVOKE SELECT ON public.prediction_turn_self_bet FROM orion_readonly;
--
-- Verify:
--   SELECT has_table_privilege('orion_readonly', 'public.curiosity_focus_request', 'SELECT'),
--          has_table_privilege('orion_readonly', 'public.curiosity_finding_register', 'SELECT'),
--          has_table_privilege('orion_readonly', 'public.prediction_turn_self_bet', 'SELECT');
