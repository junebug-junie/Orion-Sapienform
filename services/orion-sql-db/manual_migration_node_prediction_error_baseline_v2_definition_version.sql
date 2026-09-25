-- Node prediction-error EWMA baseline v2: definition_version (2026-09-25)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_node_prediction_error_baseline_v2_definition_version.sql
--
-- Records which prediction-error formula a target's baseline was built on
-- (orion/schemas/prediction_error_definitions.py). When the live version for a
-- target's reducer_key differs, AttentionRuntimeStore.advance_node_prediction_error_baseline
-- restarts that baseline cold and folds only receipts stamped with the live
-- version. Existing rows default to 1, the version of every receipt written
-- before stamping existed -- so route_arbitration/chat_session (now v2) reset
-- on the attention runtime's first tick after this is applied, and every other
-- target is untouched.
--
-- Additive, constant default: metadata-only in Postgres 11+, no table rewrite.
-- Until it is applied the attention runtime keeps the old behaviour (no reset)
-- and logs node_prediction_error_baseline_definition_version_column_missing.

alter table substrate_node_prediction_error_baseline
    add column if not exists definition_version integer not null default 1;
