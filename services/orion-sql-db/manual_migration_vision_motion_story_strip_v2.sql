-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- Walkway story pictures of the moment (2026-10-17), on top of
-- manual_migration_vision_motion_story_v1.sql.
--
-- 1. vision_motion_story gets the strip written by orion-sql-writer
--    (app/vision_story_strip.py) for each published story:
--    evidence_strip  {version, crops:[{crop_id,t,file}|{crop_id,t,missing}], frame:{file,t,crop_id}|null,
--                     frame_luminance, scene_luminance, scene_contrast, scene_measured_on, dark,
--                     dark_threshold, saved_at[, pruned_at]}
--    frame_luminance  mean luma (0..1) of the undrawn downscaled full frame
--    scene_luminance  median over the strip's crops of each crop's median luma (0..1)
--    scene_contrast   median over the strip's crops of each crop's RMS contrast (std of luma, 0..1)
--    dark             scene_contrast below VISION_STORY_STRIP_DARK_CONTRAST (null: no picture at all)
-- 2. vision_motion_story_verdict gets cant_tell: Juniper could not judge the story
--    (no picture, too dark). Such a row has verdict NULL and never counts toward the tally.
--    A column instead of a widened CHECK: changing the CHECK would need DROP CONSTRAINT,
--    which the post-pull auto-apply refuses.
--
-- Safe for the post-pull auto-apply: ADD COLUMN only, nullable, no default, no CHECK.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_motion_story_strip_v2.sql
-- Rollback (additive): ALTER TABLE vision_motion_story DROP COLUMN evidence_strip, DROP COLUMN frame_luminance,
--   DROP COLUMN scene_luminance, DROP COLUMN scene_contrast, DROP COLUMN dark; ALTER TABLE vision_motion_story_verdict DROP COLUMN cant_tell;

ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS evidence_strip JSONB;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS frame_luminance REAL;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS scene_luminance REAL;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS scene_contrast REAL;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS dark BOOLEAN;
ALTER TABLE vision_motion_story_verdict ADD COLUMN IF NOT EXISTS cant_tell BOOLEAN;
