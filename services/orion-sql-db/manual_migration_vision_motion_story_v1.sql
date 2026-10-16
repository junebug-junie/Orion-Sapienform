-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- Walkway motion stories, slice 1 (2026-10-16).
-- Design: docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md
--
-- 1. vision_motion_story: one row per closed passage on a camera stream, written by
--    orion-sql-writer (app/vision_motion_loop.py). A plain sentence plus the numbers
--    behind it and the crop ids / boxes it rests on. No embeddings, no new identity.
--    published = false rows are kept as evidence (glimpse, standing, parked, prop,
--    edge_clipped, too_long), with drop_reason saying which.
-- 2. vision_motion_story_verdict: Juniper's spot-check of a story on the Hub
--    (right / wrong_name / wrong_motion / wrong_other). One row per story; a new
--    click replaces the old verdict.
-- 3. vision_motion_story_cursor: per stream, how far stories are closed.
--
-- Safe for the post-pull auto-apply (orion/sql_migration_drift.py auto_apply_refusals):
-- three new tables, nullable columns, indexes on the new tables only, no DO blocks,
-- no functions other than now(). Indexes are CONCURRENTLY, so the file has no BEGIN/COMMIT
-- (the post-pull applier runs it outside a transaction, statement by statement).
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_motion_story_v1.sql
-- Rollback (additive): DROP TABLE vision_motion_story_verdict; DROP TABLE vision_motion_story; DROP TABLE vision_motion_story_cursor;

CREATE TABLE IF NOT EXISTS vision_motion_story (
    story_id            TEXT PRIMARY KEY,          -- sha1(stream_id | first crop_id)[:32]
    stream_id           TEXT,
    kind                TEXT,                      -- person | dog | bicycle | stroller | vehicle
    started_at          TIMESTAMPTZ,
    ended_at            TIMESTAMPTZ,
    detection_count     INTEGER,
    used_detections     INTEGER,                   -- boxes not touching the frame edge
    edge_clipped_count  INTEGER,
    speed_class         TEXT,                      -- standing|walking|running|parked|walked|riding|pushed|creeping|driving
    speed_bh_per_sec    REAL,                      -- net travel / duration, in box heights
    net_travel_bh       REAL,
    path_bh             REAL,
    dir_lateral         SMALLINT,                  -- -1 right to left, 0, +1 left to right
    dir_depth           SMALLINT,                  -- -1 away from the house, 0, +1 toward it
    zone_path           TEXT[],
    individual_id       TEXT,                      -- only when the strict naming rule holds
    name_label          TEXT,                      -- the name at write time
    name_withheld       TEXT,                      -- two_named | edge_clipped | low_share
    companion_kinds     TEXT[],
    together_story_ids  TEXT[],
    published           BOOLEAN,                   -- false: drop_reason says why
    drop_reason         TEXT,
    clean               BOOLEAN,                   -- started after VISION_MOTION_CLEAN_SINCE
    sentence            TEXT,
    crop_ids            TEXT[],
    evidence            JSONB,                     -- [{crop_id, t, label, score, box, zone, clipped, individual_id, thumb}]
    rules_version       INTEGER,
    created_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX CONCURRENTLY IF NOT EXISTS vision_motion_story_stream_time_idx
    ON vision_motion_story (stream_id, ended_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS vision_motion_story_published_idx
    ON vision_motion_story (published, started_at DESC);

CREATE TABLE IF NOT EXISTS vision_motion_story_verdict (
    story_id        TEXT PRIMARY KEY,
    verdict         TEXT CHECK (verdict IN ('right', 'wrong_name', 'wrong_motion', 'wrong_other')),
    rules_version   INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS vision_motion_story_cursor (
    stream_id        TEXT PRIMARY KEY,
    processed_until  TIMESTAMPTZ,
    updated_at       TIMESTAMPTZ DEFAULT now()
);
