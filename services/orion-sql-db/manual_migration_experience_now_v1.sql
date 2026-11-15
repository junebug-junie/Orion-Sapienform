-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- Orion's now, integration design Part A patch 1
-- (docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, A2-A4).
--
-- * temporal_self_event gets salience (the share of the same source kind's events in the 30 days before
--   it that were strictly smaller, by the kind's declared magnitude; NULL under 20 past events or with no
--   magnitude) and surprise_nats (only where a real prediction exists). Written at insert by the chronicle.
-- * experience_shown: one row per GET /temporal-self/now (what one reader was shown). A cite is valid
--   only against a shown row.
-- * experience_use: one row per valid cite (POST /temporal-self/now/use). Invalid cites are never stored.
--
-- Single writer: orion-durable-runs. Apply BEFORE deploying orion-durable-runs (its chronicle INSERT names
-- the two new columns; without them every window fails and is retried until this is applied):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_experience_now_v1.sql
-- Check: python3 scripts/check_sql_migrations_applied.py --file manual_migration_experience_now_v1.sql
-- Additive and idempotent; re-running is safe.
BEGIN;
ALTER TABLE temporal_self_event ADD COLUMN IF NOT EXISTS salience double precision;
ALTER TABLE temporal_self_event ADD COLUMN IF NOT EXISTS surprise_nats double precision;

CREATE TABLE IF NOT EXISTS experience_shown (
    shown_id       text PRIMARY KEY,
    reader_organ   text NOT NULL,
    reader_run_ref text,
    event_ids      text[] NOT NULL DEFAULT '{}',
    since          timestamptz NOT NULL,
    shown_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_experience_shown_organ_time ON experience_shown (reader_organ, shown_at);
CREATE INDEX IF NOT EXISTS idx_experience_shown_time ON experience_shown (shown_at);

CREATE TABLE IF NOT EXISTS experience_use (
    use_id       text PRIMARY KEY,
    event_id     text NOT NULL,
    citing_organ text NOT NULL,
    citing_ref   text NOT NULL,
    shown_id     text NOT NULL,
    cited_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_experience_use_event ON experience_use (event_id);
CREATE INDEX IF NOT EXISTS idx_experience_use_time ON experience_use (cited_at);
CREATE INDEX IF NOT EXISTS idx_experience_use_shown ON experience_use (shown_id);
COMMIT;

-- Indexes on the existing, live-written event table: CONCURRENTLY, outside the transaction, so the
-- chronicle's writes are never blocked while they build.
-- The now reads "new since you last looked" by ingested_at.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_self_event_ingested ON temporal_self_event (ingested_at);
-- Salience history reads one kind's last 30 days.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_self_event_kind_time ON temporal_self_event (source_kind, occurred_at);
