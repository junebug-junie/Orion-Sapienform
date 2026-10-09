-- Node prediction-error EWMA baseline v3: last_value_observed_at (2026-09-29)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_node_prediction_error_baseline_v3_last_value_observed_at.sql
--
-- When the receipt behind last_value was written. Candidate A's staleness fade
-- (orion/attention/field_attention/candidate_precision_weighted.py) ages
-- last_value from this time. last_receipt_created_at is not a substitute: it is
-- the fetch cursor and also advances over skipped receipts (malformed, or a
-- different prediction-error definition version), so during a substrate-only
-- rollback it would keep an old reading looking seconds old.
--
-- Additive, nullable, no default: metadata-only, no table rewrite. Until it is
-- applied, or for a row not yet re-folded, the attention runtime falls back to
-- the cursor and logs node_prediction_error_baseline_observed_at_column_missing.
-- Safe in either order relative to the attention-runtime deploy (the read goes
-- through to_jsonb, so a missing column reads as NULL).

alter table substrate_node_prediction_error_baseline
    add column if not exists last_value_observed_at timestamptz;
