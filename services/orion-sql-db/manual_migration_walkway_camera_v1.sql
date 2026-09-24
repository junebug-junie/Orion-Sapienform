-- Walkway camera: individuals, rhythms, unresolved percepts, and asks.
-- docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_walkway_camera_v1.sql
-- Rollback: drop the tables below (nothing else depends on them), and
--   ALTER TABLE vision_events DROP COLUMN stream_id;  (additive, nullable)

-- vision_events is shared by every camera. Which camera a narrative is about,
-- so room readers never read a walkway (street/patio) narrative as the room.
-- NULL = written before this column existed (room cameras only). orion-sql-
-- writer also applies this at boot, since its ORM now writes the column.
ALTER TABLE vision_events ADD COLUMN IF NOT EXISTS stream_id TEXT;
CREATE INDEX IF NOT EXISTS vision_events_stream_created_idx
    ON vision_events (stream_id, created_at);

-- Raw tracked-label boxes, one row per box, written by sql-writer from
-- orion:vision:crops:sql-write. Kept 7 days (pruned by the individuals
-- reducer). The CHECK is the privacy boundary as code: a box in a no-embed
-- zone (the patio; config/vision_zones.yaml `embed: false`) can never carry
-- an embedding or a thumbnail, whatever a producer sends. The writer sets
-- zone_no_embed from the config, so a second no-embed zone is covered by the
-- same constraint without editing it.
CREATE TABLE IF NOT EXISTS vision_crop_observation (
    crop_id         TEXT PRIMARY KEY,
    observation_id  TEXT NOT NULL,
    stream_id       TEXT NOT NULL,
    camera_id       TEXT,
    artifact_id     TEXT,
    observed_at     TIMESTAMPTZ NOT NULL,
    label           TEXT NOT NULL,
    score           REAL NOT NULL,
    box_xyxy        REAL[] NOT NULL,
    zone            TEXT,
    embedding_ref   TEXT,
    embedding       REAL[],
    -- orion-vision-host's crop thumbnail, "thumb:<sha256>" (embedded crops only).
    thumb_ref       TEXT,
    zone_no_embed   BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT vision_crop_observation_no_embed_zone_keeps_nothing
        CHECK (NOT zone_no_embed OR (embedding IS NULL AND embedding_ref IS NULL AND thumb_ref IS NULL))
);
CREATE INDEX IF NOT EXISTS vision_crop_observation_stream_time_idx
    ON vision_crop_observation (stream_id, observed_at);
-- The individuals reducer's cursor is landing time, so a late crop is not skipped.
CREATE INDEX IF NOT EXISTS vision_crop_observation_stream_created_idx
    ON vision_crop_observation (stream_id, created_at);

-- An appearance cluster: "the same one again". Never a face. Only Juniper
-- names one (via orion_ask). Deleting an individual deletes its sightings.
CREATE TABLE IF NOT EXISTS vision_individual (
    individual_id       TEXT PRIMARY KEY,
    stream_id           TEXT NOT NULL,
    kind                TEXT NOT NULL,          -- detector label class: person, dog, vehicle, ...
    centroid            REAL[] NOT NULL,
    centroid_n          INTEGER NOT NULL DEFAULT 1,
    first_seen_at       TIMESTAMPTZ NOT NULL,
    last_seen_at        TIMESTAMPTZ NOT NULL,
    sighting_count      INTEGER NOT NULL DEFAULT 0,
    distinct_days       INTEGER NOT NULL DEFAULT 0,
    label               TEXT,
    labeled_at          TIMESTAMPTZ,
    label_ask_id        TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS vision_individual_stream_kind_idx
    ON vision_individual (stream_id, kind);

-- One appearance: a run of observations of one individual with no gap
-- longer than the reducer's merge gap. Kept 90 days.
CREATE TABLE IF NOT EXISTS vision_individual_sighting (
    sighting_id          TEXT PRIMARY KEY,
    individual_id        TEXT NOT NULL REFERENCES vision_individual(individual_id) ON DELETE CASCADE,
    stream_id            TEXT NOT NULL,
    zone                 TEXT,
    started_at           TIMESTAMPTZ NOT NULL,
    ended_at             TIMESTAMPTZ NOT NULL,
    dwell_sec            REAL NOT NULL DEFAULT 0,
    observation_count    INTEGER NOT NULL DEFAULT 1,
    last_box_xyxy        REAL[],
    embedding_ref        TEXT,
    evidence_ref         TEXT,
    -- Thumbnail of the latest embedded crop; the ask card shows it.
    thumb_ref            TEXT,
    zone_no_embed        BOOLEAN NOT NULL DEFAULT false,
    attention_score      REAL,
    attention_components JSONB,
    -- {zone: observation count}; a sighting spans reducer ticks, so "the zone
    -- it spent most observations in" needs the running tally, not just the last zone.
    zone_counts          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT vision_individual_sighting_no_embed_zone_keeps_nothing
        CHECK (NOT zone_no_embed OR (embedding_ref IS NULL AND thumb_ref IS NULL))
);
-- For a table created from an earlier draft of this file.
ALTER TABLE vision_individual_sighting
    ADD COLUMN IF NOT EXISTS zone_counts JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS vision_individual_sighting_stream_time_idx
    ON vision_individual_sighting (stream_id, started_at);
CREATE INDEX IF NOT EXISTS vision_individual_sighting_individual_idx
    ON vision_individual_sighting (individual_id, started_at);

-- Landing time (vision_crop_observation.created_at) processed through.
CREATE TABLE IF NOT EXISTS vision_individuals_cursor (
    stream_id        TEXT PRIMARY KEY,
    last_created_at  TIMESTAMPTZ NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Predictions about the street, and their grades (PerceptExpectationV1).
CREATE TABLE IF NOT EXISTS vision_percept_expectation (
    expectation_id    TEXT PRIMARY KEY,
    stream_id         TEXT NOT NULL,
    subject_key       TEXT NOT NULL,
    subject_label     TEXT NOT NULL,
    day_kind          TEXT NOT NULL,
    window_start      TIMESTAMPTZ NOT NULL,
    window_end        TIMESTAMPTZ NOT NULL,
    peak_minute       INTEGER NOT NULL,
    support_days      INTEGER NOT NULL,
    support_sightings INTEGER NOT NULL,
    confidence        REAL NOT NULL,
    status            TEXT NOT NULL DEFAULT 'open',   -- open | met | missed | unscorable
    emitted_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    scored_at         TIMESTAMPTZ,
    outcome_event_id  TEXT,
    UNIQUE (stream_id, subject_key, window_start)
);
CREATE INDEX IF NOT EXISTS vision_percept_expectation_status_idx
    ON vision_percept_expectation (status, window_end);

CREATE TABLE IF NOT EXISTS vision_rhythm_cursor (
    stream_id    TEXT PRIMARY KEY,
    last_run_at  TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Things Orion saw and could not name (VisionUnresolvedV1).
CREATE TABLE IF NOT EXISTS vision_unresolved (
    unresolved_id   TEXT PRIMARY KEY,
    stream_id       TEXT,
    camera_id       TEXT,
    window_id       TEXT,
    observed_at     TIMESTAMPTZ NOT NULL,
    reason          TEXT NOT NULL,
    description     TEXT NOT NULL,
    what_was_tried  JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_refs   JSONB NOT NULL DEFAULT '[]'::jsonb,
    image_ref       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS vision_unresolved_observed_idx ON vision_unresolved (observed_at);

-- Orion's questions to Juniper (OrionAskV1). The daily cap is counted from
-- this table, so a restart cannot reset it.
CREATE TABLE IF NOT EXISTS orion_ask (
    ask_id        TEXT PRIMARY KEY,
    asked_of      TEXT NOT NULL DEFAULT 'juniper',
    question      TEXT NOT NULL,
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    image_ref     TEXT,
    status        TEXT NOT NULL DEFAULT 'open',       -- open | answered | dismissed | expired
    answer        TEXT,
    answered_at   TIMESTAMPTZ,
    applied_at    TIMESTAMPTZ,                        -- when the source consumer acted on the answer
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ,
    source_kind   TEXT NOT NULL,
    source_ref    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS orion_ask_status_idx ON orion_ask (status, created_at);
-- At most one open ask per subject: never ask the same thing twice at once.
CREATE UNIQUE INDEX IF NOT EXISTS orion_ask_one_open_per_source
    ON orion_ask (source_kind, source_ref) WHERE status = 'open';
