-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- Personality graph loop (spec 2026-10-16-orion-dynamic-personality-graph-design.md, rev 2).
-- Single writer: orion-durable-runs' personality reducer
-- (services/orion-durable-runs/app/personality_*.py). Readers: the same service's
-- GET /personality/* routes (Hub panel, and the frame that chat stance, reverie, curiosity and
-- outreach render with orion/personality/cue.py).
--
-- One transaction per reducer tick writes, in order: new evidence rows, changed dispositions,
-- weight-history rows for nodes whose weight or status moved, and the per-source cursors. A
-- restart resumes from the cursors: no double fold, no dropped row.
--
-- Apply BEFORE deploying orion-durable-runs (without it the reducer logs
-- `personality_tick_failed` every tick and writes nothing; nothing else is affected):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_personality_graph_v1.sql
-- Check: python3 scripts/check_sql_migrations_applied.py --file manual_migration_personality_graph_v1.sql
-- Additive and idempotent; re-running is safe. Rollback: PERSONALITY_ENABLED=false (the tables can
-- stay); manual_migration_personality_graph_v1_rollback.sql drops them.
BEGIN;

-- A node: something Orion keeps returning to. centroid is the running mean of its evidence
-- embeddings (bge-large, 1024 floats, unit length). label is an excerpt of Orion's own words.
CREATE TABLE IF NOT EXISTS personality_disposition (
    disposition_id     text PRIMARY KEY,
    origin             text NOT NULL,
    status             text NOT NULL,
    label              text NOT NULL,
    label_similarity   double precision NOT NULL DEFAULT 0,
    centroid           real[] NOT NULL,
    centroid_n         integer NOT NULL DEFAULT 0,
    weight             double precision NOT NULL DEFAULT 0,
    evidence_count     integer NOT NULL DEFAULT 0,
    lane_counts        jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- Per-lane recent hits (habituation; faded with weight, as of updated_at).
    lane_recent        jsonb NOT NULL DEFAULT '{}'::jsonb,
    chosen_count       integer NOT NULL DEFAULT 0,
    first_seen_at      timestamptz NOT NULL,
    last_reinforced_at timestamptz NOT NULL,
    retired_at         timestamptz,
    updated_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_personality_disposition_status_weight
    ON personality_disposition (status, weight DESC);

-- One piece of evidence: a lane output or a curiosity choice, and the node it reinforced.
CREATE TABLE IF NOT EXISTS personality_evidence (
    evidence_id    text PRIMARY KEY,
    lane           text NOT NULL,
    source_ref     text NOT NULL,
    occurred_at    timestamptz NOT NULL,
    excerpt        text NOT NULL,
    disposition_id text,
    similarity     double precision,
    weight_delta   double precision NOT NULL DEFAULT 0,
    meta           jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_personality_evidence_disposition
    ON personality_evidence (disposition_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_personality_evidence_lane_time
    ON personality_evidence (lane, occurred_at);

-- A node's weight over time (written only when weight moved >= 1% or status changed).
CREATE TABLE IF NOT EXISTS personality_weight_history (
    disposition_id text NOT NULL,
    at             timestamptz NOT NULL,
    weight         double precision NOT NULL,
    status         text NOT NULL,
    evidence_count integer NOT NULL,
    PRIMARY KEY (disposition_id, at)
);

-- Per-source cursors. A source with no row starts at the reducer's first boot (no backfill:
-- spec rev 2 starts the graph from today).
CREATE TABLE IF NOT EXISTS personality_cursor (
    source         text PRIMARY KEY,
    cursor_at      timestamptz NOT NULL,
    cursor_ref     text,
    updated_at     timestamptz NOT NULL DEFAULT now()
);

COMMIT;
