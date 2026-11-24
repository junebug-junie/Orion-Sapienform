-- ORION-MIGRATION-REQUIRED-BY: orion-substrate-runtime
-- Signal history for the five substrate nodes that are NOT prediction errors (design
-- docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md, PR #3781, Patch 1).
--
-- kind 'alarm':        node:substrate.cabinet, node:substrate.strain, node:substrate.street
-- kind 'scene_change': node:substrate.perception, node:substrate.perception_walkway
--
-- Single writer: orion-substrate-runtime (orion/substrate/signal_history.py). In Patch 1 it
-- DUAL-WRITES: the same reading still goes to substrate_node_prediction_error_history, and also
-- lands here. Same shape and PK as that table, so a repeat write is a no-op. Pruned by the writer
-- at SUBSTRATE_PE_HISTORY_RETENTION_HOURS (default 168), like the PE table. No reader uses it yet.
--
-- Without it the service still runs: the dual-write fails open and logs
-- substrate_signal_history_write_failed.
--
-- Apply (the post-pull rebuild applies it automatically: additive DDL only):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_node_signal_history_v1.sql
-- Check: python3 scripts/check_sql_migrations_applied.py --file manual_migration_node_signal_history_v1.sql
-- No GRANT: the postgres role's default privileges already give orion_readonly SELECT on new tables.
-- Additive and idempotent; re-running is safe.

CREATE TABLE IF NOT EXISTS substrate_node_signal_history (
    node_id      text             NOT NULL,
    observed_at  timestamptz      NOT NULL,
    value        double precision NOT NULL,
    kind         text             NOT NULL CHECK (kind IN ('alarm', 'scene_change')),
    recorded_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (node_id, observed_at)
);
-- Retention prune and the window reads filter on observed_at alone.
CREATE INDEX IF NOT EXISTS idx_node_signal_history_observed
    ON substrate_node_signal_history (observed_at);
