-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Splitting walkway individuals from Juniper's picture answers (2026-10-16).
-- Design: docs/superpowers/specs/2026-10-16-vision-individual-split-proposal.md
--
-- 1. vision_crop_observation.individual_id: the individual the clusterer put
--    this crop in. Written by orion-sql-writer for every assigned crop from
--    this migration on. NULL for every older row (no backfill here) and for
--    crops that never become an individual (no embedding, no-embed zone).
-- 2. vision_individual_split: one row per split a "someone else" picture
--    verdict caused (or refused / skipped, with the reason), with the exact
--    before/after state so scripts/undo_vision_split.py can put it back.
--    It holds crop ids and centroids, never pictures.
--
-- Safe for the post-pull auto-apply (orion/sql_migration_drift.py
-- auto_apply_refusals): a nullable column with no default or constraint on
-- the existing table, a CONCURRENTLY index on the existing table (so this
-- file has no BEGIN/COMMIT), and a new table with its own indexes.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_split_v1.sql
-- Rollback (additive): DROP TABLE vision_individual_split; DROP INDEX CONCURRENTLY vision_crop_observation_individual_idx; ALTER TABLE vision_crop_observation DROP COLUMN individual_id;

ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS individual_id TEXT;

CREATE INDEX CONCURRENTLY IF NOT EXISTS vision_crop_observation_individual_idx
    ON vision_crop_observation (individual_id, observed_at);

CREATE TABLE IF NOT EXISTS vision_individual_split (
    split_id                    TEXT PRIMARY KEY,
    source_individual_id        TEXT NOT NULL,      -- the individual that was split (it keeps its id, label, node)
    new_individual_id           TEXT,               -- created by the split; NULL when refused / skipped
    ask_id                      TEXT NOT NULL,      -- the answered "who is this?" card
    group_key                   TEXT NOT NULL,      -- lower(name), or '' for unnamed "someone else" pictures
    name                        TEXT,               -- the typed name, if any
    status                      TEXT NOT NULL CHECK (status IN ('split', 'refused', 'skipped')),
    reason                      TEXT,               -- why refused / skipped
    moved                       JSONB NOT NULL DEFAULT '[]'::jsonb,  -- [{crop_id, moved_by, prior_individual_id}]
    before                      JSONB,              -- source row + its affected sighting rows, in full
    after                       JSONB,              -- the same after the split, plus the new individual
    linked_pair                 JSONB NOT NULL DEFAULT '[]'::jsonb,  -- pair verdict rows the split wrote
    sightings_out_of_retention  INTEGER NOT NULL DEFAULT 0,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    undone_at                   TIMESTAMPTZ,
    UNIQUE (source_individual_id, ask_id, group_key)
);
CREATE INDEX CONCURRENTLY IF NOT EXISTS vision_individual_split_new_idx
    ON vision_individual_split (new_individual_id);
