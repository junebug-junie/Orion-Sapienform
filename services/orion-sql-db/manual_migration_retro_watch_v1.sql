-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs
-- Game company, patch 1 (2026-11-17, docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md
-- section 1): a sitting is one stretch of play by a person on the Retro service.
--
-- retro_sitting: one row per sitting. Hub's Retro gateway writes it (services/orion-hub/scripts/
-- retro_sitting.py): upserted once a minute while the sitting is open and once more at its close.
-- backfilled = true rows come from scripts/backfill_retro_sittings.py (the evidence journals); they
-- are a baseline only and never become Temporal Self events. Readers: Hub (the game page and the
-- chat prompt read the open sitting from Redis, not from here), the backfill's own report.
--
-- retro_play_sessions.player_stated: what the game page was told ('juniper' or 'guest'), NULL when
-- nobody said. Written only on kind = 'human_play' rows; read by the Temporal Self source
-- `retro_play` (services/orion-durable-runs/app/temporal_self_sources.py) for the arc's label.
-- No name is ever guessed or copied from a camera.
--
-- Additive only: one new table, one index, one nullable column.
-- Rollback: DROP TABLE retro_sitting; ALTER TABLE retro_play_sessions DROP COLUMN player_stated.
CREATE TABLE IF NOT EXISTS retro_sitting (
    sitting_id          text PRIMARY KEY,
    retro_session_id    text NOT NULL,
    game_id             text,
    game_title          text,
    started_at          timestamptz NOT NULL,
    last_input_at       timestamptz NOT NULL,
    ended_at            timestamptz,
    end_reason          text,
    mask_changes        integer NOT NULL DEFAULT 0,
    saves               integer NOT NULL DEFAULT 0,
    loads               integer NOT NULL DEFAULT 0,
    pauses              integer NOT NULL DEFAULT 0,
    player_stated       text,
    backfilled          boolean NOT NULL DEFAULT false,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS retro_sitting_started_idx ON retro_sitting (started_at DESC);
ALTER TABLE retro_play_sessions ADD COLUMN IF NOT EXISTS player_stated text;
