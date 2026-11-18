-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs
-- Game company, patch 3 (2026-11-18, docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md
-- revision 4, sections 4, 5 and 8): Orion speaks while a person plays.
--
-- retro_line: one row per line turn Hub (hub-app, services/orion-hub/scripts/retro_commentary.py) started for a
-- funded bid of family hub.retro_commentary. The row is written BEFORE the turn runs with outcome 'running'.
-- Juniper's message marks a running line 'yielded' (UPDATE ... WHERE outcome = 'running') before the turn is
-- killed, and delivery claims the row the same way (running -> delivered), so a yielded line can never be said,
-- even when its draft finishes later. Outcomes: delivered | passed (Orion chose silence) | yielded | no_frames (no
-- ring: the browser closed or reconnected) | failed. Seconds by stage (stance, motor, finalize) are read from the
-- turn's own cockpit hops; lease_priorities are the GPU pool priorities the line asked for (its hold and the route
-- every stage attaches with). thumb / replied_*: did the player like it; thumb_by 'guest' is stored and never
-- counted as Juniper's. text is Orion's own line (no prompt): the frames are not stored, only their hashes.
--
-- retro_watch_look gains the judgment of patch 3: a line turn shown a note settles its bucket. checkable is true
-- when the note's frame window overlapped the turn's frames; verdict moves to checked_holds or contradicted only
-- for a checkable note the turn judged. settled_at is when that happened. Only checked_holds looks become
-- Temporal Self events (source kind retro_look, orion-durable-runs), keyed by settled_at.
--
-- Additive only. The index is on the table this file creates. Safe to apply twice.
-- Rollback: DROP TABLE retro_line; ALTER TABLE retro_watch_look DROP COLUMN checkable, DROP COLUMN settled_at,
--   DROP COLUMN settled_by_line.
ALTER TABLE retro_watch_look ADD COLUMN IF NOT EXISTS checkable boolean;
ALTER TABLE retro_watch_look ADD COLUMN IF NOT EXISTS settled_at timestamptz;
ALTER TABLE retro_watch_look ADD COLUMN IF NOT EXISTS settled_by_line text;

CREATE TABLE IF NOT EXISTS retro_line (
    line_id             text PRIMARY KEY,
    sitting_id          text NOT NULL,
    game_id             text,
    want_id             text NOT NULL,
    correlation_id      text NOT NULL,
    decided_at          timestamptz NOT NULL,
    finished_at         timestamptz,
    outcome             text NOT NULL CHECK (outcome IN ('running', 'delivered', 'passed', 'yielded', 'no_frames', 'failed')),
    failure             text,
    text                text,
    look_ids            text[] NOT NULL DEFAULT '{}',
    frame_hashes        text[] NOT NULL DEFAULT '{}',
    frames_expire_at    timestamptz,
    now_cites           text[] NOT NULL DEFAULT '{}',
    looks_shown         integer NOT NULL DEFAULT 0,
    looks_checkable     integer NOT NULL DEFAULT 0,
    looks_contradicted  integer NOT NULL DEFAULT 0,
    stance_sec          double precision,
    motor_sec           double precision,
    finalize_sec        double precision,
    total_sec           double precision,
    hold_lease_id       text,
    lease_priorities    text[] NOT NULL DEFAULT '{}',
    player_stated       text,
    message_id          text,
    thumb               text CHECK (thumb IN ('up', 'down')),
    thumb_by            text CHECK (thumb_by IN ('juniper', 'guest')),
    thumb_at            timestamptz,
    replied_at          timestamptz,
    replied_by          text CHECK (replied_by IN ('juniper', 'guest')),
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS retro_line_sitting_idx ON retro_line (sitting_id, decided_at);
