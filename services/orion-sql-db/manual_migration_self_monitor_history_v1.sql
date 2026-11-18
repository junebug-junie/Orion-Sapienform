-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs
-- Self-monitor organ, patch 2 (host and publish; docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md).
-- Single writer: orion-durable-runs, the `self_monitor` node of the temporal_self.update thread. It
-- appends one row per hourly evaluation (the whole SelfMonitorStateV1 it published to Redis
-- orion:self_monitor:latest) and prunes rows older than SELF_MONITOR_HISTORY_RETENTION_DAYS (30).
-- Faculty status changes go to the existing temporal_self_event table (kind faculty_status_changed),
-- which needs no change.
--
-- Without it the node still publishes Redis and writes its status-change rows, but logs
-- `self_monitor_history_write_failed` and the hourly history is lost.
-- Apply (the post-pull rebuild applies it automatically: additive DDL only):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_self_monitor_history_v1.sql
-- Check: python3 scripts/check_sql_migrations_applied.py --file manual_migration_self_monitor_history_v1.sql
-- Additive and idempotent; re-running is safe.
BEGIN;
CREATE TABLE IF NOT EXISTS substrate_self_monitor_history (
    computed_at  timestamptz PRIMARY KEY,
    state_json   jsonb NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);
COMMIT;
