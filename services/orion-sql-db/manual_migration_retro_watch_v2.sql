-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs
-- Game company, patch 2 (2026-11-18, docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md
-- revision 4, sections 3, 6 part 8 and 8): Orion looks at the game screen while a person plays.
--
-- Writer of every table here: Hub's Retro gateway (services/orion-hub/scripts/retro_watch.py).
--
-- retro_sitting (patch 1) gains the look counters, in the order every report gives them: ring checks,
-- checks over the floor, checks skipped for stale input, checks under the floor with input, bids funded,
-- model calls, notes stored; failed looks last. static_skips counts checks under the floor.
--
-- retro_play_sessions.looks: notes Orion stored while watching a human_play sitting. Read by the Temporal
-- Self source retro_play (services/orion-durable-runs/app/temporal_self_sources.py) for the arc's label
-- ("... Orion watched: N notes"). NULL on every other row.
--
-- retro_watch_look: one row per look that got a GPU seat. A bid the market refused, or a seat the pool
-- refused, writes no row (the gateway logs it and the page says why). verdict is 'unchecked' until patch 3's
-- line turn judges the note; only 'checked_holds' looks ever become temporal events (patch 3). The frames
-- themselves live one hour in orion-percept-store; frame_hashes outlive them, frames_expire_at says when
-- they went (last post + 1 h). The note is a look, never a fact.
--
-- retro_watch_change_hist: every 2-second ring check of every game, aggregated (never one row per check):
-- the count of checks per (game, a button changed or was held in the last 30 s, bin of the screen change
-- against the check 30 s before). Bin k of 82, on a log scale of 20 bins a decade from 1e-4 to 1
-- (orion/retro/watch.py EDGES).
--
-- retro_watch_floor: one row per game, the newest calibration from that histogram. floor is NULL until a
-- game has >= 30 minutes of checks and one value separates still from played checks (<= 10% / >= 90%).
-- histogram holds the counts the result came from. Nothing is seeded by hand.
--
-- retro_watch_preference: Juniper's switch on the game page (set_by 'juniper', scope 'all'), and Orion's
-- choices (set_by 'orion', scope 'sitting' or 'game'), copied by the gateway from Orion's own act rows
-- (grammar_acts, actor orion.watch_choice, written by `dig watch`). The lower of the two sides wins.
--
-- Additive only. The two indexes are on tables this file creates. Safe to apply twice.
-- Rollback: DROP TABLE retro_watch_look, retro_watch_change_hist, retro_watch_floor, retro_watch_preference;
--   ALTER TABLE retro_play_sessions DROP COLUMN looks; ALTER TABLE retro_sitting DROP COLUMN <each counter>.
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS ring_checks integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS checks_over_floor integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS input_stale_skips integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS under_floor_with_input integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS static_skips integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS bids_funded integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS model_calls integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS notes_stored integer NOT NULL DEFAULT 0;
ALTER TABLE retro_sitting ADD COLUMN IF NOT EXISTS failed_looks integer NOT NULL DEFAULT 0;
ALTER TABLE retro_play_sessions ADD COLUMN IF NOT EXISTS looks integer;

CREATE TABLE IF NOT EXISTS retro_watch_look (
    look_id              text PRIMARY KEY,
    sitting_id           text NOT NULL,
    game_id              text,
    looked_at            timestamptz NOT NULL,
    frame_hashes         text[] NOT NULL DEFAULT '{}',
    frames_expire_at     timestamptz,
    note                 text,
    outcome              text NOT NULL CHECK (outcome IN ('stored', 'empty', 'unclear', 'copied', 'failed')),
    failure              text,
    verdict              text NOT NULL DEFAULT 'unchecked' CHECK (verdict IN ('unchecked', 'checked_holds', 'contradicted')),
    screen_change        double precision,
    want_id              text NOT NULL,
    lease_id             text,
    lane_wait_ms         integer,
    seconds              double precision,
    prompt_tokens        integer,
    similar_to_previous  double precision,
    created_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS retro_watch_look_sitting_idx ON retro_watch_look (sitting_id, looked_at);

CREATE TABLE IF NOT EXISTS retro_watch_change_hist (
    game_id          text NOT NULL,
    input_in_window  boolean NOT NULL,
    bin              smallint NOT NULL,
    checks           bigint NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, input_in_window, bin)
);

CREATE TABLE IF NOT EXISTS retro_watch_floor (
    game_id          text PRIMARY KEY,
    floor            double precision,
    status           text NOT NULL,
    checks_input     integer NOT NULL,
    checks_no_input  integer NOT NULL,
    seconds          double precision NOT NULL,
    noinput_above    double precision,
    input_above      double precision,
    histogram        jsonb NOT NULL,
    computed_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retro_watch_preference (
    pref_id          text PRIMARY KEY,
    scope            text NOT NULL CHECK (scope IN ('sitting', 'game', 'all')),
    scope_id         text NOT NULL DEFAULT '',
    set_by           text NOT NULL CHECK (set_by IN ('orion', 'juniper')),
    mode             text NOT NULL CHECK (mode IN ('off', 'quiet', 'on')),
    why              text,
    act_id           text,
    correlation_id   text,
    set_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS retro_watch_preference_side_idx ON retro_watch_preference (set_by, scope, scope_id, set_at DESC);
