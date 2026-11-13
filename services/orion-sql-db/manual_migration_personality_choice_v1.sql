-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- Personality rev 4: choices in (spec docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md,
-- rev 4 "one organ: choices in, values out").
--
-- personality_choice: one row per PersonalityChoiceV1 (orion/schemas/personality.py) -- what Orion
-- picked at a choice point, and the options Orion turned down. Insert-only (choice_id is the
-- producer's deterministic id; a re-delivered choice is a no-op). Single writer: orion-durable-runs,
-- from bus orion:personality:choice and from its own curiosity sync (curiosity_offer_decisions +
-- curiosity_run_outcomes, read-only). Reader: the same service's personality reducer, which folds
-- ONLY these rows into personality_disposition.
--
-- personality_disposition.opposed_count: choices whose direction went against a node (the label's
-- "n of m"). Defaulted: additive only.
--
-- Apply BEFORE deploying orion-durable-runs (the deploy gate refuses it until then):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_personality_choice_v1.sql
-- Rollback: manual_migration_personality_choice_v1_rollback.sql (drops the table and the column).

CREATE TABLE IF NOT EXISTS personality_choice (
    choice_id           text PRIMARY KEY,
    lane                text NOT NULL,
    occurred_at         timestamptz NOT NULL,
    -- {key, text, market_rank}; NULL when declined_all.
    chosen              jsonb,
    -- [{key, text, market_rank}], at least one.
    alternatives        jsonb NOT NULL,
    chosen_market_rank  integer,
    declined_all        boolean NOT NULL DEFAULT false,
    care_promoted       boolean NOT NULL DEFAULT false,
    picks_in_decision   integer NOT NULL DEFAULT 1,
    situation           jsonb NOT NULL DEFAULT '{}'::jsonb,
    act_id              text,
    correlation_id      text,
    -- bus | curiosity_sync
    written_by          text NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS personality_choice_created_idx
    ON personality_choice (created_at, choice_id);
CREATE INDEX IF NOT EXISTS personality_choice_lane_time_idx
    ON personality_choice (lane, occurred_at);
-- The curiosity sync reads a run only while it has no row here (one decision, synced once).
CREATE INDEX IF NOT EXISTS personality_choice_act_idx
    ON personality_choice (act_id);

ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS opposed_count integer NOT NULL DEFAULT 0;
