-- Per-node prediction-error reading history (spec: docs/superpowers/specs/
-- 2026-10-02-reverie-prediction-error-magnitude-proposal.md, step 1).
--
-- One row per real observed_at advance of a node:substrate.* node's
-- prediction_error, written by orion-substrate-runtime's attention-broadcast
-- tick when SUBSTRATE_PE_HISTORY_ENABLED=true. Stale nodes (observed_at
-- unchanged) add no rows: the PK makes a repeat write a no-op. Feeds
-- orion/substrate/prediction_error_magnitude.py (7d/24h percentiles, trend),
-- attached to broadcast loops as OpenLoopV1.magnitude. Pruned by the writer at
-- SUBSTRATE_PE_HISTORY_RETENTION_HOURS (default 168). Expected volume: under
-- 30k rows/day. Purely additive; dropping it costs nothing but the history.
--
-- Apply BEFORE setting SUBSTRATE_PE_HISTORY_ENABLED=true:
--   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_node_prediction_error_history_v1.sql

create table if not exists substrate_node_prediction_error_history (
    node_id text not null,
    observed_at timestamptz not null,
    value double precision not null,
    recorded_at timestamptz not null default now(),
    primary key (node_id, observed_at)
);
-- Retention prune and the 7-day seed read both filter on observed_at alone.
create index if not exists idx_node_pe_history_observed
    on substrate_node_prediction_error_history(observed_at);
