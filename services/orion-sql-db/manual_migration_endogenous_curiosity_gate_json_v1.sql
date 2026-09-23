-- Additive gate lineage for System One curiosity admission.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_curiosity_gate_json_v1.sql
-- Safe to re-run. Does not alter candidates_json shape used by felt-state readers.

alter table substrate_endogenous_curiosity_candidates
    add column if not exists gate_json jsonb;

comment on column substrate_endogenous_curiosity_candidates.gate_json is
    'Optional System One curiosity admission provenance for this candidate set '
    '(frame_id, probabilities, argmax level, gate_result, evaluator outcome).';
