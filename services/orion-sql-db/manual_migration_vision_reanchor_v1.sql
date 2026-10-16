-- ORION-MIGRATION-REQUIRED-BY: none -- audit table for the operator script scripts/reanchor_vision_individuals.py only; no service reads or writes it
-- Re-anchoring named walkway individuals' body centroids (2026-10-16).
-- Design: docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "Re-anchor".
--
-- vision_individual_reanchor: one row per individual the re-anchor script
-- rewrote (action 'reanchor') or put back from a snapshot (action 'undo'),
-- with the exact before/after centroid and centroid_n. It holds crop ids and
-- body centroids, never pictures. (vision_individual_split does not fit: its
-- status CHECK and NOT NULL ask_id are split-specific.)
--
-- Safe for the post-pull auto-apply (orion/sql_migration_drift.py
-- auto_apply_refusals): a new table and an index on that new table only.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_reanchor_v1.sql
-- Rollback (additive): DROP TABLE vision_individual_reanchor;

CREATE TABLE IF NOT EXISTS vision_individual_reanchor (
    reanchor_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,          -- one script run (one transaction)
    individual_id   TEXT NOT NULL,
    action          TEXT NOT NULL CHECK (action IN ('reanchor', 'undo')),
    rule            TEXT NOT NULL,          -- which crops built the new centroid
    crop_ids        JSONB NOT NULL DEFAULT '[]'::jsonb,
    before          JSONB NOT NULL,         -- {centroid, centroid_n}
    after           JSONB NOT NULL,         -- {centroid, centroid_n}
    snapshot_path   TEXT,                   -- /tmp/vision-reanchor/<ts>/before.json
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS vision_individual_reanchor_individual_idx
    ON vision_individual_reanchor (individual_id, created_at);
