-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- Visual confirmation asks (2026-10-15): Orion asks Juniper to confirm what the
-- walkway camera saw, with pictures.
--
-- 1. orion_ask.image_refs: every picture an ask shows (a "same person?" card
--    shows two faces). image_ref stays for old rows and old readers; new rows
--    set image_ref = image_refs[0].
-- 2. vision_individual_pair_verdict: Juniper's answer to "are these two the
--    same person?". verdict='same' links the two individuals (nothing is
--    merged or deleted); verdict='different' means the pair is never asked
--    again. face_cosine is the similarity of the two face centroids when the
--    answer was applied: each row is one labeled calibration point for
--    VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD. No foreign key on purpose: the
--    verdict outlives the 90-day individual prune (ids only, no pictures).
--
-- Every step is idempotent. Requires manual_migration_walkway_camera_v1.sql
-- and manual_migration_walkway_privacy_v2.sql.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql
-- Rollback (additive): DROP TABLE vision_individual_pair_verdict; ALTER TABLE orion_ask DROP COLUMN image_refs;

ALTER TABLE orion_ask ADD COLUMN IF NOT EXISTS image_refs JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS vision_individual_pair_verdict (
    individual_a  TEXT NOT NULL,                 -- the smaller id of the pair
    individual_b  TEXT NOT NULL,                 -- the larger id of the pair
    verdict       TEXT NOT NULL CHECK (verdict IN ('same', 'different')),
    face_cosine   REAL,                          -- face-centroid similarity when applied; NULL if a centroid was gone
    ask_id        TEXT NOT NULL,
    decided_at    TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (individual_a, individual_b),
    CHECK (individual_a < individual_b)
);
CREATE INDEX IF NOT EXISTS vision_individual_pair_verdict_b_idx
    ON vision_individual_pair_verdict (individual_b);
