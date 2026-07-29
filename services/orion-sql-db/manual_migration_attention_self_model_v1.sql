-- Append-only store for AttentionSelfModelV1 (AST/HOT reducer output),
-- one row per live self-model tick.
--
-- reduce_attention_self_model() (orion/substrate/attention_self_model.py)
-- was pure and correct, but until this table's writer shipped its only
-- real caller was _brain_frame_tick() (services/orion-substrate-runtime/
-- app/worker.py) -- a narrower computation (field_frame=None, no trend)
-- embedded inline into one brain-frame UI dimension, never persisted
-- durably. This table (docs/superpowers/specs/2026-07-29-ast-hot-reducer-
-- live-ticking-design.md) is the first durable home for the complete
-- self-model (real field lane + real trend). Review finding 2026-07-29:
-- the two live computations are intentionally left un-unified -- see
-- _attention_self_model_tick()'s own docstring for the full reasoning.
--
-- Deliberately a brand-new table with exactly one writer
-- (services/orion-substrate-runtime/app/worker.py's
-- _attention_broadcast_tick(), appended tail call, gated behind
-- SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED) -- this repo has a
-- documented, repeated failure class of a periodic tick clobbering a field
-- it doesn't exclusively own (execution_load cross-lane stomp PR #1338,
-- orion-field-digester's generic decay clobber, SubstrateDynamicsEngine.
-- tick()'s bus_synaptic clobber PR #1449). A table nothing else has ever
-- written to makes that failure class structurally impossible here, not
-- just avoided by convention.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_self_model_v1.sql

create table if not exists substrate_attention_self_model (
    model_id text primary key,
    generated_at timestamptz not null,
    self_model_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_attention_self_model_generated
    on substrate_attention_self_model(generated_at desc);
