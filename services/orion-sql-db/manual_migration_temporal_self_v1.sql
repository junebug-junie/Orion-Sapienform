-- Temporal Self rev 4 (PR #2369), patch 3: the live chronology's tables.
-- Single writer: orion-durable-runs, the `chronicle` node of the temporal_self.update thread
-- (services/orion-durable-runs/app/temporal_self_chronicle.py). Readers: the same service's
-- GET /temporal-self/* routes. temporal_self_event already exists
-- (manual_migration_temporal_self_event_v1.sql, applied 2026-10-10); this file only adds a column.
--
-- One transaction per chronicle window writes, in order: new events, changed arcs, closed days,
-- the current-day frame, the per-source cursors, and the reducer state (its own checkpoint, so a
-- restart resumes exactly where the last committed window ended: no double fold, no dropped row).
--
-- Apply BEFORE deploying orion-durable-runs (without it the chronicle node logs
-- `temporal_self_chronicle_failed` every step and writes nothing; regulation is unaffected):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_temporal_self_v1.sql
-- Check: python3 scripts/check_sql_migrations_applied.py --file manual_migration_temporal_self_v1.sql
-- Additive and idempotent; re-running is safe. Rollback: TEMPORAL_SELF_CHRONICLE_ENABLED=false
-- (the tables can stay; nothing else reads them).
BEGIN;

-- A row that arrived after its window was read: stored so it is counted once, never folded.
ALTER TABLE temporal_self_event ADD COLUMN IF NOT EXISTS late_unfolded boolean NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS temporal_self_arc (
    arc_id            text PRIMARY KEY,
    day_id            text NOT NULL,
    kind              text NOT NULL,
    subject_ref       text NOT NULL,
    began_at          timestamptz NOT NULL,
    ended_at          timestamptz,
    status            text NOT NULL,
    attention_returns integer NOT NULL DEFAULT 0,
    arc_json          jsonb NOT NULL,
    updated_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_temporal_self_arc_day ON temporal_self_arc (day_id, began_at);
CREATE INDEX IF NOT EXISTS idx_temporal_self_arc_subject ON temporal_self_arc (subject_ref, day_id);

-- A closed local day: its final frame and every arc in full (outlives arc retention).
CREATE TABLE IF NOT EXISTS temporal_self_day (
    day_id    text PRIMARY KEY,
    closed_at timestamptz NOT NULL,
    day_json  jsonb NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now()
);

-- Singleton 'current_day': the latest TemporalSelfFrameV1, verbatim.
CREATE TABLE IF NOT EXISTS temporal_self_projection (
    projection_id   text PRIMARY KEY,
    generated_at    timestamptz NOT NULL,
    projection_json jsonb NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Per-source cursors: the last folded row of each source, plus 'read_watermark' (every source is
-- read up to it). Observability; the restart point is temporal_self_state.
CREATE TABLE IF NOT EXISTS temporal_self_cursor (
    source_kind      text PRIMARY KEY,
    last_occurred_at timestamptz,
    last_source_ref  text,
    updated_at       timestamptz NOT NULL DEFAULT now()
);

-- The reducer's own checkpoint (TemporalSelfStateV1 JSON, gzipped: ~1.2 MB raw at a busy midday).
CREATE TABLE IF NOT EXISTS temporal_self_state (
    state_id        text PRIMARY KEY,
    watermark       timestamptz NOT NULL,
    -- Where reading began (first boot or a reset). Nothing before it was ever read, so the
    -- late-row probe never reaches before it.
    origin          timestamptz NOT NULL,
    reducer_version text NOT NULL,
    state_gz        bytea NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- The attention_row read is by generated_at every step; the table had no index on it
-- (149k rows live on 10-10, seq-scanned twice per 2-minute step without this).
CREATE INDEX IF NOT EXISTS idx_substrate_attention_schema_generated_at ON substrate_attention_schema (generated_at);

COMMIT;
