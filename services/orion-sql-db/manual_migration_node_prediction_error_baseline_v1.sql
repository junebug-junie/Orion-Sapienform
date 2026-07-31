-- Node prediction-error EWMA baseline v1 (2026-07-30 fix)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_node_prediction_error_baseline_v1.sql
--
-- Persisted, incrementally-updated running baseline for Candidate A's five
-- node:substrate.* precision-weighted-salience targets
-- (orion/attention/field_attention/selectors.py::PREDICTION_ERROR_NATIVE_TARGETS).
-- Replaces AttentionRuntimeStore.load_prediction_error_history's per-tick
-- ~30-minute rolling-window recompute -- see
-- orion/attention/field_attention/candidate_precision_weighted.py's module
-- docstring and orion/sentience_striving_program/README.md section 12 for the
-- live incident this fixes. One row per target_id, advanced by
-- AttentionRuntimeStore.advance_node_prediction_error_baseline (one real
-- substrate_reduction_receipts row = one real update, not a per-tick recompute).

create table if not exists substrate_node_prediction_error_baseline (
    target_id text primary key,
    reducer_key text not null,
    ewma double precision not null default 0.0,
    variance double precision not null default 0.0,
    observation_count integer not null default 0,
    last_value double precision,
    last_receipt_created_at timestamptz,
    updated_at timestamptz not null default now()
);

create index if not exists idx_substrate_node_prediction_error_baseline_updated_at
    on substrate_node_prediction_error_baseline (updated_at desc);
