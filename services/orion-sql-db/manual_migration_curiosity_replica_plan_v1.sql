-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Curiosity replica, patch 2 (contracts): the plan table and the offer row's arm propensities.
-- Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md, sections 4-6 and
-- "Proposed schema / API changes".
--
-- curiosity_offer_decisions.arm_propensities: the chance each offer arm had for this run, as
--   {"value_order": p, "replica_order": p, "uncertainty_order": p} (sums to 1). Written by Hub
--   from orion/curiosity/value.py draw_offer_arm on every offer row from this patch on
--   (services/orion-hub/scripts/curiosity_offer_decisions.py record_offer_decision), so Hub needs
--   the column before it starts. NULL on rows written before it: read those from
--   value_arm_propensity (the two-arm draw, P(replica) = 0). value_arm_propensity stays written
--   for old readers.
--
-- curiosity_offer_plans: one row per replica-arm run (run_id), whatever happened. Writer: Hub
--   (patch 3, the plan consumer); no writer in this patch, so the table stays empty until then.
--   status: 'ok' | 'abstain' | 'unfunded' | 'model_failed' | 'unparseable' (orion-mind's answer,
--   orion/schemas/curiosity_replica.py CuriosityOfferPlanResultV1) | 'timeout' | 'invalid:<reason>'
--   (Hub's own verdict). fallback_reason is NULL only when the plan was used (status 'ok'); every
--   other row is a fallback to the uncertainty order and counts against the replica arm
--   (intention to treat, design section 5).
--   picks: [{"prior_id", "why"}] in the replica's order. The "why" text is the replica's reasons:
--   they stay here and never reach Orion's prompt.
--
-- Deliberately no GRANT to orion_readonly (the FCC sandbox): an in-turn sandbox must not read the
-- replica's reasons (design section 5), the same rule as the spend log. Hub connects as the table
-- owner, so it needs no grant.
--
-- Additive DDL only, so the post-pull applier (scripts/apply_post_pull_migrations.py) applies it.
-- Apply by hand (idempotent):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_curiosity_replica_plan_v1.sql

ALTER TABLE curiosity_offer_decisions ADD COLUMN IF NOT EXISTS arm_propensities JSONB;

CREATE TABLE IF NOT EXISTS curiosity_offer_plans (
    run_id text PRIMARY KEY,
    plan_id text,
    requested_at timestamptz NOT NULL DEFAULT now(),
    answered_at timestamptz,
    status text NOT NULL,
    fallback_reason text,
    picks jsonb NOT NULL DEFAULT '[]'::jsonb,
    focus_stance jsonb,
    signal_stance jsonb,
    act_id text,
    latency_ms integer,
    frame_tokens integer,
    CHECK (status IN ('ok', 'abstain', 'unfunded', 'model_failed', 'unparseable', 'timeout')
           OR status LIKE 'invalid:%'),
    CHECK ((status = 'ok') = (fallback_reason IS NULL))
);

CREATE INDEX IF NOT EXISTS curiosity_offer_plans_requested_at
    ON curiosity_offer_plans (requested_at);
