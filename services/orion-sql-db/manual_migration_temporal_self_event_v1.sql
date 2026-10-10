-- Temporal Self rev 4 (PR #2369), order 3 "R2/R3 core": the event table only.
-- Single writer: orion-durable-runs, the `regulate` node of the temporal_self.update thread,
-- which writes one `arousal_transition` row per arousal level change (spec R3 "History").
-- Patch 3 (Temporal Self nodes) adds temporal_self_arc / _day / _projection / _cursor in its
-- own migration and reuses this table unchanged (same columns as the spec's
-- manual_migration_temporal_self_v1.sql block).
--
-- Apply BEFORE deploying orion-durable-runs (without it the service still runs and still writes
-- Redis orion:regulation:latest, but logs `regulation_transition_write_failed` and the history
-- is lost):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_temporal_self_event_v1.sql
-- Check: python3 scripts/check_sql_migrations_applied.py --file manual_migration_temporal_self_event_v1.sql
-- Additive and idempotent; re-running is safe.
BEGIN;
CREATE TABLE IF NOT EXISTS temporal_self_event (
    event_id       text PRIMARY KEY,
    day_id         text NOT NULL,
    occurred_at    timestamptz NOT NULL,
    source_kind    text NOT NULL,
    source_table   text,
    source_ref     text,
    correlation_id text,
    subject_ref    text,
    related_refs   text[] NOT NULL DEFAULT '{}',
    label          text,
    verdict        text,
    payload_json   jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_temporal_self_event_day ON temporal_self_event (day_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_temporal_self_event_time ON temporal_self_event (occurred_at, event_id);
COMMIT;
