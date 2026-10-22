-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs
-- Orion plays Zelda (2026-10-22, docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md
-- section F): one row per finished Retro session (co-play or leisure). Hub writes it
-- (services/orion-hub/scripts/retro_memory_export.py, which also creates it lazily with a short
-- lock_timeout); the Temporal Self chronicle reads it as source `retro_play`
-- (services/orion-durable-runs/app/temporal_self_sources.py).
-- Facts and counts only: no frames, RAM, ROM or raw journal.
-- Additive only: one new table and one index. Rollback: DROP TABLE retro_play_sessions.
CREATE TABLE IF NOT EXISTS retro_play_sessions (
    session_id          text PRIMARY KEY,
    lineage_id          text,
    game_id             text NOT NULL,
    game_title          text NOT NULL,
    kind                text NOT NULL,
    dispatch_id         text,
    started_at          timestamptz NOT NULL,
    ended_at            timestamptz NOT NULL,
    duration_sec        double precision NOT NULL,
    end_reason          text,
    control             jsonb NOT NULL DEFAULT '[]'::jsonb,
    human_frames        integer NOT NULL,
    orion_frames        integer NOT NULL,
    human_share         double precision,
    played_together     boolean NOT NULL,
    rooms_new           integer NOT NULL,
    caves_new           integer NOT NULL,
    items_gained        jsonb NOT NULL DEFAULT '[]'::jsonb,
    deaths              integer NOT NULL,
    death_rooms         jsonb NOT NULL DEFAULT '[]'::jsonb,
    decisions           integer NOT NULL,
    predictions_met     integer NOT NULL,
    predictions_refuted integer NOT NULL,
    predictions_unknown integer NOT NULL,
    human_interventions integer NOT NULL,
    summary             text NOT NULL,
    memory_card_id      uuid,
    card_error          text,
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS retro_play_sessions_ended_idx ON retro_play_sessions (ended_at DESC);
