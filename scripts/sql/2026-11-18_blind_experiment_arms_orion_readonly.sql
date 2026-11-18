-- Hide blinded experiment arms from Orion's read-only SQL role, at the database (issue #3458).
--
-- WHY. Two experiments are blind to Orion until they stop:
--   * the three-arm curiosity test (#3183): offer arms value_order / replica_order / uncertainty_order
--     (orion/curiosity/value.py OFFER_ARMS);
--   * the dream hypothesis test: arms dream / control (orion/schemas/dream_cycle.py HypothesisArm).
-- Patch 1 (#3453) took the arm out of the now's labels. The arm stayed readable by SQL as `orion_readonly`,
-- the role behind ORION_CURIOSITY_PG_DSN in every turn (orion/curiosity/sandbox_env.py) and behind the dig kit's
-- ReadOnlyPsql (orion/evals/model_replay/sandbox.py). Orion's turns run raw psql, so a code-level check in one
-- runner cannot close this; the role's own privileges are the boundary.
--
-- WHAT orion_readonly COULD READ (checked live 2026-11-18, read-only, as postgres):
--   curiosity_offer_decisions.arm   NOT granted (has_table_privilege = f). Pinned below with a no-op REVOKE.
--   dream_hypothesis.arm            NOT granted (f). Pinned below.
--   dream_cycle.cycle_json          NOT granted (f). Pinned below (dream BLIND_FORBIDDEN_COLUMNS).
--   temporal_self_event.payload_json  GRANTED (2026-11-16_grant_orion_readonly_organ_roundup.sql). Top-level key
--                                   `arm` on every curiosity_run row (216) and every dream_hypothesis row (179).
--                                   No other source kind carries "arm"; no label, subject_ref, related_refs or
--                                   verdict carries an arm name.
--   pg_stats on temporal_self_event  READABLE: the planner's most-common payload_json values. pg_stats shows a
--                                   column only to a role that can SELECT it, so the REVOKE below closes this too.
-- A text search for the three offer-arm names over every other table orion_readonly can read found only
-- gpu_pool's unrelated `reason=value_order` and design-doc prose; no per-run assignment.
--
-- HOW. The same pattern as scope_orion_readonly_off_aitown.sql: a view of the same name in schema orion_scope,
-- which is first on this role's search_path, so `SELECT ... FROM temporal_self_event` (the dig kit's organs and
-- attention queries) keeps working unchanged. The view drops the top-level `arm` key from payload_json. One
-- exception: a curiosity_run row whose arm is `self_inquiry` keeps it. self_inquiry is a line, not an offer arm
-- (no draw), and the now already names it ("self-inquiry run"). Any other arm, on any source kind, now or later,
-- is hidden by default. Then the table grant goes away, so `public.temporal_self_event` is refused.
-- Views run with their owner's privileges, so the view needs no grant on the table.
--
-- WHAT THIS DOES NOT COVER. It hides the STORED arm from orion_readonly. It does not make the arm unknowable:
--   * #3685: the curiosity arm is a sha256 of the run id (orion/curiosity/value.py draw_offer_arm), and the run id,
--     the formula and the propensities are all visible to Orion. Needs Juniper's decision.
--   * #3680: a turn can reach Postgres as superuser through the harness governor's docker socket
--     (`docker exec orion-athena-sql-db psql -U postgres`, local trust).
--   * Row order: dream hypotheses were stored dream arm first, then control, with one occurred_at per cycle, so
--     scan order told the arm for rows written before this PR (43 of 44 mixed cycles, checked 2026-11-18).
--     orion-durable-runs now stores each window in event_id order (window_event_rows). Old rows keep their order
--     until the table is rewritten (CLUSTER or VACUUM FULL, a locking production operation; not done here).
--
-- Not a migration: GRANT/REVOKE/CREATE VIEW are not on the post-pull auto-apply allowlist. Apply by hand.
-- Idempotent; safe to re-run. Do NOT re-run the temporal_self_event GRANT line of
-- 2026-11-16_grant_orion_readonly_organ_roundup.sql while this is in force: it reopens the leak.
--
-- Apply (from the host, primary checkout on main):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-18_blind_experiment_arms_orion_readonly.sql
--
-- If a later migration adds a column to temporal_self_event, add it to the view below and re-apply
-- (tests/test_blind_experiment_arms_sql.py fails until you do).

BEGIN;

SET LOCAL lock_timeout = '10s';

CREATE SCHEMA IF NOT EXISTS orion_scope;
GRANT USAGE ON SCHEMA orion_scope TO orion_readonly;

CREATE OR REPLACE VIEW orion_scope.temporal_self_event AS
  SELECT e.event_id, e.day_id, e.occurred_at, e.source_kind, e.source_table, e.source_ref, e.correlation_id,
         e.subject_ref, e.related_refs, e.label, e.verdict,
         CASE WHEN e.source_kind = 'curiosity_run' AND e.payload_json ->> 'arm' = 'self_inquiry'
              THEN e.payload_json
              ELSE e.payload_json - 'arm'
         END AS payload_json,
         e.ingested_at, e.late_unfolded, e.salience, e.surprise_nats
  FROM public.temporal_self_event e;

GRANT SELECT ON orion_scope.temporal_self_event TO orion_readonly;

-- The revoke is what makes the view a boundary rather than a suggestion.
REVOKE SELECT ON public.temporal_self_event FROM orion_readonly;

-- Already not granted live; pinned so a stray grant cannot reopen the source tables.
REVOKE SELECT ON public.curiosity_offer_decisions FROM orion_readonly;
REVOKE SELECT ON public.dream_hypothesis FROM orion_readonly;
REVOKE SELECT ON public.dream_cycle FROM orion_readonly;

-- Same value as live (scope_orion_readonly_off_aitown.sql); restated so this file stands alone.
ALTER ROLE orion_readonly SET search_path = orion_scope, public;

COMMIT;

-- Verify (as postgres, after apply):
--   SELECT has_table_privilege('orion_readonly', 'public.temporal_self_event', 'SELECT');        -- expect f
--   SELECT has_table_privilege('orion_readonly', 'orion_scope.temporal_self_event', 'SELECT');   -- expect t
--   SELECT has_table_privilege('orion_readonly', 'public.curiosity_offer_decisions', 'SELECT');  -- expect f
--   SELECT has_table_privilege('orion_readonly', 'public.dream_hypothesis', 'SELECT');           -- expect f
--   BEGIN; SET LOCAL ROLE orion_readonly; SET LOCAL search_path = orion_scope, public;
--   SELECT count(*) FROM temporal_self_event
--    WHERE source_kind IN ('curiosity_run', 'dream_hypothesis')
--      AND payload_json ? 'arm' AND payload_json ->> 'arm' <> 'self_inquiry';                   -- expect 0
--   SELECT count(*) FROM public.temporal_self_event;                                             -- expect DENIED
--   ROLLBACK;
-- Then `python -P -m orion.dig organs <date>` still prints experiences, not a GRANT line.
--
-- LIFT, when an experiment stops. The column lists below must match the main view; if a column was added since,
-- add it here too (CREATE OR REPLACE VIEW refuses to drop a column, so a stale block fails loudly, not silently). The curiosity test stops at 250 completed runs per arm or 180 days
-- (scripts/analysis/curiosity_three_arm_readout.py prints this file's name once it reads "Stopped"; #3183).
-- The dream hypothesis test has no stop rule yet.
--
--   Curiosity stopped, dream still blind: re-create the view with the exception widened to every curiosity_run:
--     CREATE OR REPLACE VIEW orion_scope.temporal_self_event AS
--       SELECT e.event_id, e.day_id, e.occurred_at, e.source_kind, e.source_table, e.source_ref, e.correlation_id,
--              e.subject_ref, e.related_refs, e.label, e.verdict,
--              CASE WHEN e.source_kind = 'curiosity_run' THEN e.payload_json ELSE e.payload_json - 'arm' END
--                AS payload_json,
--              e.ingested_at, e.late_unfolded, e.salience, e.surprise_nats
--       FROM public.temporal_self_event e;
--
--   Both stopped, full undo (the role reads the table again; the source-table REVOKEs stay, they were never granted):
--     BEGIN;
--     GRANT SELECT ON public.temporal_self_event TO orion_readonly;
--     DROP VIEW orion_scope.temporal_self_event;
--     COMMIT;
