-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Motion stories: per-camera frame size, entry/exit edges, exit speed (2026-10-20),
-- on top of manual_migration_vision_motion_story_v1.sql and _strip_v2.sql.
-- Design: docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md, section 1.
--
-- 1. vision_crop_observation gets the frame size the detector saw (from the
--    vision.crop.observation.v1 payload's frame_width/frame_height, which the
--    writer used to drop). NULL on rows written before this migration.
-- 2. vision_motion_story gets, per story (NULL on stories written before it):
--    entry_edge / exit_edge  'left' | 'right' | 'top' | 'bottom' | 'inside': where the
--                            first / last box sat. A box within VISION_MOTION_EDGE_ZONE_FRAC
--                            (default 0.05) of the frame from the left, right or bottom
--                            edge is on that edge (nearest wins); 'top' only when the FOOT
--                            point is that close to the top; else 'inside' (began or ended
--                            mid-frame: a door, a stop, the detector lost it).
--    exit_foot_x / exit_foot_y  the last box's foot point (bottom centre), 0..1 of its frame.
--    exit_speed_fw_per_sec   lateral foot speed over the last <= 4 unclipped boxes, frame
--                            widths per second, UNSIGNED (direction: exit_edge, dir_lateral).
--                            NULL with fewer than 2 unclipped boxes.
--    frame_w / frame_h       the frame size the story was measured in (pixels).
--    frame_size_source       'detection' (the crops' own frame size) | 'stream_setting'
--                            (VISION_MOTION_FRAME_SIZE_BY_STREAM) | 'default' (global setting).
--
-- Safe for the post-pull auto-apply: ADD COLUMN only, nullable, no default, no CHECK.
-- The writer tolerates this file being absent (it logs and writes the older columns).
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_motion_story_edges_v1.sql
-- Rollback (additive): ALTER TABLE vision_crop_observation DROP COLUMN frame_width, DROP COLUMN frame_height;
--   ALTER TABLE vision_motion_story DROP COLUMN entry_edge, DROP COLUMN exit_edge, DROP COLUMN exit_foot_x,
--   DROP COLUMN exit_foot_y, DROP COLUMN exit_speed_fw_per_sec, DROP COLUMN frame_w, DROP COLUMN frame_h,
--   DROP COLUMN frame_size_source;

ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS frame_width INTEGER;
ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS frame_height INTEGER;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS entry_edge TEXT;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS exit_edge TEXT;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS exit_foot_x REAL;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS exit_foot_y REAL;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS exit_speed_fw_per_sec REAL;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS frame_w INTEGER;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS frame_h INTEGER;
ALTER TABLE vision_motion_story ADD COLUMN IF NOT EXISTS frame_size_source TEXT;
