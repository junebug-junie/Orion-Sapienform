-- Append-only store for CodebaseMassBaseline (codebase-mass domain's
-- three-domain EWMA state -- orion/substrate/prediction_error.py's
-- codebase_prediction_error()), one row per real consumer tick.
--
-- Same shape as manual_migration_attention_self_model_v1.sql, and for the
-- identical reason: this domain's EWMA baseline has no persisted reducer
-- projection to live on (unlike the other five domains in
-- orion/substrate/prediction_error.py, each of which stores its own
-- baseline directly on a projection object) -- see
-- docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md's
-- "Producer + consumer patch design" section for the full reasoning,
-- including why the originally-proposed alternative (piggyback on
-- node:substrate.codebase's own ConceptNodeV1.metadata) was verified wrong:
-- services/orion-substrate-runtime/app/worker.py::_write_prediction_error_node()
-- builds a fresh metadata dict every call, carrying forward only a narrow,
-- dynamics-engine-owned key allowlist -- a custom baseline key there would
-- be silently dropped on the very next write.
--
-- Deliberately a brand-new table with exactly one writer
-- (services/orion-substrate-runtime/app/worker.py's
-- _codebase_delta_listener_loop(), gated behind
-- SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE) -- this repo has a
-- documented, repeated failure class of a periodic tick clobbering a field
-- it doesn't exclusively own (execution_load cross-lane stomp PR #1338,
-- orion-field-digester's generic decay clobber, SubstrateDynamicsEngine.
-- tick()'s bus_synaptic clobber PR #1449). A table nothing else has ever
-- written to makes that failure class structurally impossible here, not
-- just avoided by convention.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_codebase_mass_baseline_v1.sql

create table if not exists substrate_codebase_mass_baseline (
    baseline_id text primary key,
    generated_at timestamptz not null,
    baseline_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_codebase_mass_baseline_generated
    on substrate_codebase_mass_baseline(generated_at desc);
