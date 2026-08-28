-- Take AI Town out of what Orion can SELECT, at the database, not in a prompt.
--
-- WHY. `orion/curiosity/study_material.py` filters AI Town out of the twelve
-- cards Orion is shown, but the curiosity sandbox also carries
-- ORION_CURIOSITY_PG_DSN -- role `orion_readonly` -- and the prompt hands Orion
-- that DSN and names `memory_crystallizations` directly. So the menu was clean
-- and the pantry was not: run `e537ad51afa2` on 2026-08-28 went straight to
-- psql, read the raw table, found 31 AI Town roleplay crystallizations from
-- July 31 and reasoned about them. Juniper: "but why is it referencing the ai
-- town bullshit at all". Because nothing stopped it.
--
-- `sandbox_env.py` already states the principle this restores: "What the
-- credentials THEMSELVES allow is the real boundary, and it is enforced by
-- Postgres and FalkorDB, not by this file." That was true for writes and never
-- true for AI Town scope.
--
-- THE PROMPT DOES NOT CHANGE. The views live in their own schema and
-- `search_path` puts that schema first for this role only, so Orion's existing
-- `SELECT ... FROM memory_crystallizations` resolves to the filtered view. A
-- rename would have meant every run that used the old name burning budget on a
-- permission error, and the prompt naming tables that exist for nobody else.
--
-- Of the four tables `orion_readonly` can read, only two leak:
--   chat_history_log                   already AI-Town-free by construction --
--                                      PR #1734 routes those rows to
--                                      `aitown_chat_history_log`, which this
--                                      role has no grant on at all
--   journal_entries                    Orion's own writing
--   memory_crystallizations            796 of 1291 rows are AI Town
--   memory_concept_relation_decisions  judgements about those rows
--
-- Measured live 2026-08-28 in a rolled-back transaction: 1291 rows visible
-- before, 495 after, and `SELECT ... FROM public.memory_crystallizations`
-- returns "permission denied".
--
-- Idempotent. Safe to re-run.
--
-- TO UNDO, in full:
--   ALTER ROLE orion_readonly RESET search_path;
--   GRANT SELECT ON public.memory_crystallizations,
--                   public.memory_concept_relation_decisions TO orion_readonly;
--   DROP SCHEMA orion_scope CASCADE;

BEGIN;

CREATE SCHEMA IF NOT EXISTS orion_scope;

-- A crystallization is AI Town's if ANY of its sources is an AI Town turn.
-- `aitown_chat_history_log` is the signal rather than a platform tag on this
-- table, because that table is AI-Town-only BY CONSTRUCTION since the #1734
-- split -- the same reasoning `concept_atlas_routes` already relies on.
CREATE OR REPLACE VIEW orion_scope.memory_crystallizations AS
  SELECT m.*
  FROM public.memory_crystallizations m
  WHERE NOT EXISTS (
    SELECT 1
    FROM public.memory_crystallization_sources s
    JOIN public.aitown_chat_history_log a ON a.id::text = s.source_id
    WHERE s.crystallization_id = m.crystallization_id
  );

-- Either end being AI Town's disqualifies the decision. The candidate side is
-- stored `crys_<hex-no-dashes>` and the target as a dashed uuid -- the same id
-- space in two formats, which is why the candidate half of induction read as
-- 0/550 dangling until it was normalised.
CREATE OR REPLACE VIEW orion_scope.memory_concept_relation_decisions AS
  SELECT d.*
  FROM public.memory_concept_relation_decisions d
  WHERE NOT EXISTS (
    SELECT 1
    FROM public.memory_crystallization_sources s
    JOIN public.aitown_chat_history_log a ON a.id::text = s.source_id
    WHERE ('crys_' || replace(s.crystallization_id::text, '-', ''))
            = d.candidate_crystallization_id
       OR s.crystallization_id::text = d.target_crystallization_id
  );

GRANT USAGE ON SCHEMA orion_scope TO orion_readonly;
GRANT SELECT ON orion_scope.memory_crystallizations TO orion_readonly;
GRANT SELECT ON orion_scope.memory_concept_relation_decisions TO orion_readonly;

-- The revoke is what makes the view a boundary rather than a suggestion.
-- Views run with their OWNER's privileges, so dropping these grants does not
-- break the views themselves.
REVOKE SELECT ON public.memory_crystallizations FROM orion_readonly;
REVOKE SELECT ON public.memory_concept_relation_decisions FROM orion_readonly;

-- Unqualified names resolve to the filtered views for this role only. `public`
-- stays on the path so the other two granted tables still resolve.
ALTER ROLE orion_readonly SET search_path = orion_scope, public;

COMMIT;

-- Verification, run as yourself afterwards:
--
--   SET ROLE orion_readonly; SET search_path = orion_scope, public;
--   SELECT count(*) FROM memory_crystallizations;          -- expect 495
--   SELECT count(*) FROM public.memory_crystallizations;   -- expect DENIED
--   RESET ROLE;
