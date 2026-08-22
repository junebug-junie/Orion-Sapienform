-- Per-window scene census for object permanence.
--
-- Written on EVERY window, unlike vision_events: the council only emits an
-- event when the observed label set changes (reason=stable_scene otherwise),
-- so a pure count change produces nothing, and a departure is a non-event by
-- nature. Permanence needs a continuous record, so it gets its own table.
--
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS vision_scene_inventory (
    window_id        TEXT PRIMARY KEY,
    stream_id        TEXT,
    camera_id        TEXT,
    observed_at      TIMESTAMPTZ NOT NULL,
    window_start_ts  DOUBLE PRECISION,
    window_end_ts    DOUBLE PRECISION,
    frame_count      INTEGER NOT NULL DEFAULT 0,
    counts           JSONB NOT NULL DEFAULT '{}'::jsonb,
    detections       JSONB NOT NULL DEFAULT '{}'::jsonb,
    believed_labels  JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The sweep's only query shape: "per stream, newest first, within a window of
-- time". Without this it degenerates to a full scan, and this table gains a
-- row every ~5s per camera.
CREATE INDEX IF NOT EXISTS vision_scene_inventory_stream_observed_idx
    ON vision_scene_inventory (stream_id, observed_at DESC);

-- "when did I last see label X" without scanning every row. GIN on the counts
-- object supports `counts ? 'chair'` containment.
CREATE INDEX IF NOT EXISTS vision_scene_inventory_counts_gin_idx
    ON vision_scene_inventory USING GIN (counts);

-- NOTE: no retention policy ships with this table, and it grows at roughly one
-- row per window per camera (~17k/day/camera at the current 5s cadence).
-- Retention is deliberately left for the sweep patch, which is the thing that
-- will know how much history the reducer actually needs. Do not let this sit
-- unbounded -- see the substrate retention work for how that goes.
