-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Office sessions (2026-11-05, office camera design PR #3127 step B): one row per visit to a room
-- camera (cam0, the office). A visit is one presence stretch from vision_presence_transition
-- (present .. recent/absent/unknown). The row holds who was probably there and an ordered timeline
-- of what they did (the window's camera_caption rows, consecutive repeats folded).
--
-- Writer: orion-sql-writer (app/vision_office_sessions.py, VISION_OFFICE_SESSIONS_ENABLED). Without
-- this table the loop logs one ERROR and backs off; nothing else changes.
-- Readers: orion-durable-runs, the Temporal Self chronicle's `office_session` source (labels the
-- office `company` arc). It skips the read while the table is missing.
--
-- Columns:
--   session_id     "<stream>-<transition_id of the opening 'present' row>"; re-running is a no-op.
--   status         open (the visit is going on, or ended less than VISION_OFFICE_SESSIONS_SETTLE_SEC
--                  ago) | closed (final: never rewritten after this, except ask_id).
--   ended_at       the last sighting (the 'recent' row, or the absent/unknown row); NULL while open.
--   end_reason     left | camera_dark | producer_restart | stale (no presence row for 1500 s: producer gone).
--   closed_at      when the row became final; the Temporal Self reads it at this time.
--   who            {"text", "people": [{"name","basis","individual_ids"}], "unknown", "peak_people", ...}
--   timeline       [{"at","text","repeats","event_id"}], oldest first.
--   image_refs     person crop thumbnails of the visit (unknown visitors first). Thumbs are kept forever
--                  by orion-vision-host, so the visit keeps its pictures after sightings are pruned.
--   ask_id         the "Who is this?" ask (orion_ask, vision_individual) opened for an unknown person.
-- Retention: none (one row per visit, a few a day).
-- Additive and idempotent; re-running is safe.
CREATE TABLE IF NOT EXISTS vision_office_session (
    session_id      text PRIMARY KEY,
    stream_id       text NOT NULL,
    status          text NOT NULL DEFAULT 'open',
    started_at      timestamptz NOT NULL,
    last_seen_at    timestamptz NOT NULL,
    ended_at        timestamptz,
    end_reason      text,
    closed_at       timestamptz,
    who_text        text NOT NULL DEFAULT '',
    who             jsonb NOT NULL DEFAULT '{}'::jsonb,
    unknown_people  integer NOT NULL DEFAULT 0,
    timeline        jsonb NOT NULL DEFAULT '[]'::jsonb,
    label           text NOT NULL DEFAULT '',
    individual_ids  jsonb NOT NULL DEFAULT '[]'::jsonb,
    image_refs      jsonb NOT NULL DEFAULT '[]'::jsonb,
    evidence_refs   jsonb NOT NULL DEFAULT '[]'::jsonb,
    ask_id          text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT vision_office_session_status CHECK (status IN ('open', 'closed')),
    CONSTRAINT vision_office_session_closed CHECK (status = 'open' OR (closed_at IS NOT NULL AND ended_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS vision_office_session_stream_started_idx ON vision_office_session (stream_id, started_at);
CREATE INDEX IF NOT EXISTS vision_office_session_closed_idx ON vision_office_session (closed_at);
