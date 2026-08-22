-- Object permanence: persisted per-(stream, label) inventory state, updated
-- by a timer-driven sweep, separate from the per-frame vision_scene_inventory
-- writes. A departure is a non-event -- nothing fires when a thing stops
-- being there -- so this table exists to be read by a clock, not a trigger.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_vision_object_permanence_v1.sql

CREATE TABLE IF NOT EXISTS vision_object_inventory (
    stream_id       TEXT NOT NULL,
    label           TEXT NOT NULL,
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL,
    last_count      INTEGER NOT NULL DEFAULT 1,
    state           TEXT NOT NULL DEFAULT 'present',   -- 'present' | 'departed'
    state_since     TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, label)
);

-- The sweep's own query shape on every tick: "everything currently tracked
-- for this stream". Without this it is a full scan of a table that only
-- grows.
CREATE INDEX IF NOT EXISTS vision_object_inventory_stream_idx
    ON vision_object_inventory (stream_id);

-- The sweep bookkeeping table: when did the last sweep actually run, per
-- stream, so the next one knows how far back to read vision_scene_inventory
-- from. A single global cursor would be wrong the first time a second stream
-- (carbon) starts later than the first (cam0) and needs its own catch-up.
CREATE TABLE IF NOT EXISTS vision_object_permanence_cursor (
    stream_id       TEXT PRIMARY KEY,
    last_swept_at   TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- NOTE: no retention policy ships with vision_object_inventory. It grows by
-- at most (labels x streams) rows -- bounded by the detector vocabulary, not
-- by time -- so unlike vision_scene_inventory this is not an unbounded-growth
-- risk on its own. Worth revisiting if 'departed' rows are ever purged.
