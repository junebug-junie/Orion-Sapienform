-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Orion's Day corrections: a claim in a letter that a conversation with Juniper showed was
-- wrong, recorded by Orion, and offered to exactly one later curiosity run as evidence.
-- Design: docs/superpowers/specs/2026-10-11-orion-day-letter-reread-design.md, section 4.
--
-- Writers (all Hub, all in orion/orion_day/corrections.py):
--   * the record_correction introspect responder INSERTs one row per correction
--     (services/orion-hub/scripts/orion_day_correction_listener.py). correction_id is a
--     uuid5 of (ref, claim, correction, chat_correlation_id), so a retried tool call
--     lands on ON CONFLICT DO NOTHING instead of a second row.
--   * the curiosity kickoff stamps offered_at / offered_run_id (TAKE ... FOR UPDATE SKIP
--     LOCKED, the carry-forward shape) and clears them again if the run was cancelled
--     before Orion saw the prompt.
--   * the curiosity turn's end stamps priors_retested / priors_moved / outcome_stamped_at.
-- Nothing else is ever updated: the correction's own words are append-only.
--
-- This table never changes a belief. Priors live in FalkorDB (orion_worldview) and change
-- only through Orion's own writes inside a curiosity run (the PR #2199 boundary); Hub only
-- READS the worldview to validate prior_ids and to stamp what the run did.
--
-- priors_retested: named priors the offered run stamped (last_run_id = offered_run_id) and
--   whose times_tested went up. A named prior the run never touched is absent.
-- priors_moved: named priors that run moved (confidence changed, or closed as refuted /
--   retired_unresolvable). NULL = unknown (no turn-start snapshot to compare against),
--   which is different from '{}' (compared, nothing moved).
-- outcome_stamped_at NULL on an offered row = the run's end was never scored.
--
-- Grants: orion-hub connects as `postgres` (POSTGRES_URI, checked 2026-09-30), which owns
-- this table, so no GRANT is needed. No grant to orion_readonly (the FCC sandbox).
--
-- Apply manually (idempotent):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_orion_day_correction_v1.sql

CREATE TABLE IF NOT EXISTS orion_day_correction (
    correction_id uuid PRIMARY KEY,
    letter_date date NOT NULL,
    ref text NOT NULL,
    claim text NOT NULL,
    correction text NOT NULL,
    evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
    prior_ids text[] NOT NULL DEFAULT '{}',
    chat_correlation_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    offered_at timestamptz,
    offered_run_id text,
    priors_retested text[],
    priors_moved text[],
    outcome_stamped_at timestamptz,
    CHECK (length(btrim(claim)) > 0),
    CHECK (length(btrim(correction)) > 0),
    CHECK ((offered_at IS NULL) = (offered_run_id IS NULL))
);

-- The kickoff takes the oldest unoffered corrections.
CREATE INDEX IF NOT EXISTS orion_day_correction_unoffered
    ON orion_day_correction (created_at) WHERE offered_at IS NULL;
-- The turn's end and the cancel path find a run's corrections.
CREATE INDEX IF NOT EXISTS orion_day_correction_offered_run
    ON orion_day_correction (offered_run_id) WHERE offered_run_id IS NOT NULL;
-- The next letter's gather reads corrections created in its window.
CREATE INDEX IF NOT EXISTS orion_day_correction_created_at
    ON orion_day_correction (created_at);
