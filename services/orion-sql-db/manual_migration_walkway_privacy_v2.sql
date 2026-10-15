-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Walkway camera v2 (2026-10-15): patio fingerprinting on, faces for anyone on
-- the walkway, stationary fixtures.
--
-- Juniper lifted the walkway privacy limits (Utah: home cameras not pointed
-- into homes are lawful; the walkway is public). config/vision_zones.yaml now
-- has patio `embed: true`, so patio boxes carry embeddings and thumbnails and
-- the CHECKs that forbade that are dropped. zone_no_embed stays (the writer
-- still sets it for any future no-embed zone and strips vectors itself).
--
-- New nullable columns:
--   vision_crop_observation.face_embedding/face_score/face_thumb_ref -- the
--     facenet face inside a walkway person crop (orion-vision-host want_crop_faces).
--   vision_individual.face_centroid/face_n -- face centroid, preferred for assignment.
--   vision_individual.still_box/still_since/still_last/stationary and
--   vision_individual_sighting.stationary -- a box that has not moved for
--     VISION_STATIONARY_MIN_SEC is a prop (the patio skeleton), not a visitor.
--   vision_individual_sighting.face_thumb_ref -- the ask card prefers it.
--
-- Every step is idempotent. Requires manual_migration_walkway_camera_v1.sql.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_walkway_privacy_v2.sql
-- Rollback (additive columns; the CHECKs only come back if no row breaks them):
--   ALTER TABLE vision_crop_observation DROP COLUMN face_embedding, DROP COLUMN face_score, DROP COLUMN face_thumb_ref;
--   ALTER TABLE vision_individual DROP COLUMN face_centroid, DROP COLUMN face_n, DROP COLUMN still_box,
--     DROP COLUMN still_since, DROP COLUMN still_last, DROP COLUMN stationary;
--   ALTER TABLE vision_individual_sighting DROP COLUMN face_thumb_ref, DROP COLUMN stationary;

ALTER TABLE vision_crop_observation
    DROP CONSTRAINT IF EXISTS vision_crop_observation_no_embed_zone_keeps_nothing;
ALTER TABLE vision_individual_sighting
    DROP CONSTRAINT IF EXISTS vision_individual_sighting_no_embed_zone_keeps_nothing;
-- The name-based CHECKs from an earlier draft of v1, in case one survived.
ALTER TABLE vision_crop_observation
    DROP CONSTRAINT IF EXISTS vision_crop_observation_patio_no_embedding;
ALTER TABLE vision_individual_sighting
    DROP CONSTRAINT IF EXISTS vision_individual_sighting_patio_no_embedding;

ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS face_embedding REAL[];
ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS face_score REAL;
ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS face_thumb_ref TEXT;

ALTER TABLE vision_individual ADD COLUMN IF NOT EXISTS face_centroid REAL[];
ALTER TABLE vision_individual ADD COLUMN IF NOT EXISTS face_n INTEGER NOT NULL DEFAULT 0;
ALTER TABLE vision_individual ADD COLUMN IF NOT EXISTS still_box REAL[];
ALTER TABLE vision_individual ADD COLUMN IF NOT EXISTS still_since TIMESTAMPTZ;
ALTER TABLE vision_individual ADD COLUMN IF NOT EXISTS still_last TIMESTAMPTZ;
ALTER TABLE vision_individual ADD COLUMN IF NOT EXISTS stationary BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE vision_individual_sighting ADD COLUMN IF NOT EXISTS face_thumb_ref TEXT;
ALTER TABLE vision_individual_sighting ADD COLUMN IF NOT EXISTS stationary BOOLEAN NOT NULL DEFAULT false;
