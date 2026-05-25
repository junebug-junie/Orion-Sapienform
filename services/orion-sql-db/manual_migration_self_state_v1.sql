-- Self-state v1 (apply before orion-self-state-runtime)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_self_state_v1.sql

create table if not exists substrate_self_state (
    self_state_id text primary key,
    source_field_tick_id text not null,
    source_attention_frame_id text not null,
    generated_at timestamptz not null,
    policy_id text not null,
    self_state_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_self_state_generated_at
    on substrate_self_state (generated_at desc);

create index if not exists idx_substrate_self_state_source_field_tick
    on substrate_self_state (source_field_tick_id);

create index if not exists idx_substrate_self_state_source_attention_frame
    on substrate_self_state (source_attention_frame_id);
