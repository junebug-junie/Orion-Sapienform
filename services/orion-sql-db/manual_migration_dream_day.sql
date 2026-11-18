-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- The day dream (2026-11-18, docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md, Patch 2).
-- Contract: orion/schemas/dream_day.py (DreamDayV1 is one row of this table).
--
-- One row per Denver day: the night's `dream.day` durable run read that day and wrote what it kept.
-- Single writer: services/orion-durable-runs/app/dream_day_store.py, from the run's `persist` node, in ONE
-- transaction with the run's episode_memory rows (channel 'dream'):
--   INSERT ... ON CONFLICT (day) DO NOTHING  -- first run to persist wins; a later run writes nothing.
-- Readers: the chronicle (temporal_self_sources.py, source kind dream_day: one experience per row),
-- services/orion-durable-runs/evals/run_dream_day_eval.py, and Patch 3's letter.
--
-- plan holds the parsed, ref-checked keep/drop/new plan for Prime's concerns. Patch 2a only STORES it;
-- nothing hands it to Prime yet (Patch 2b).
--
-- Additive and idempotent (post-pull auto-apply allowlist: create table / create index only).
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_dream_day.sql
-- Rollback: DROP TABLE dream_day;  (the memories are episode_memory rows with channel 'dream' and this run_id)

CREATE TABLE IF NOT EXISTS dream_day (
    day date PRIMARY KEY,
    run_id text NOT NULL,
    status text NOT NULL DEFAULT 'written',
    produced boolean NOT NULL,
    note text NOT NULL DEFAULT '',
    memories_written integer NOT NULL,
    memory_ids text[] NOT NULL DEFAULT '{}',
    items_dropped integer NOT NULL DEFAULT 0,
    refs_dropped integer NOT NULL DEFAULT 0,
    memories_rejected integer NOT NULL DEFAULT 0,
    plan_rejected integer NOT NULL DEFAULT 0,
    kept integer NOT NULL DEFAULT 0,
    dropped integer NOT NULL DEFAULT 0,
    new integer NOT NULL DEFAULT 0,
    plan jsonb NOT NULL DEFAULT '[]'::jsonb,
    day_text text,
    day_refs text[] NOT NULL DEFAULT '{}',
    material jsonb NOT NULL DEFAULT '{}'::jsonb,
    read jsonb,
    now_shown_id text,
    turn_correlation_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dream_day_created_at_idx ON dream_day (created_at);
CREATE INDEX IF NOT EXISTS dream_day_run_id_idx ON dream_day (run_id);
