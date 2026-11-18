-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Hub attention minutes (2026-11-10): one row per (minute, client) in which Juniper was actually
-- looking at the Hub -- the page was visible, the window had focus, and there was real input in the
-- last 120 s. Writer: orion-hub POST /api/hub-attention/heartbeat
-- (services/orion-hub/scripts/hub_attention_routes.py, HUB_ATTENTION_TRACKER_ENABLED).
-- `minute` is date_trunc('minute', now()) on the server; client clocks are never trusted.
-- No reader yet: the prediction service's reachable forecast reads this table later, after ~2 weeks
-- of live data pass the metric gate. Without this migration the Hub logs and drops each heartbeat.
-- Additive only: one new table and one index. Rollback: DROP TABLE hub_attention_minute.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_hub_attention_minute_v1.sql

CREATE TABLE IF NOT EXISTS hub_attention_minute (
    minute    timestamptz NOT NULL,
    client    text        NOT NULL,
    tab       text,
    beats     integer     NOT NULL DEFAULT 1,
    first_at  timestamptz NOT NULL DEFAULT now(),
    last_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (minute, client)
);

CREATE INDEX IF NOT EXISTS hub_attention_minute_last_at_idx
    ON hub_attention_minute (last_at DESC);
