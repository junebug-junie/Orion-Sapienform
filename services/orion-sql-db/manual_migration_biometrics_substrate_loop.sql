-- Biometrics substrate closed loop (apply before orion-substrate-runtime)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql

create table if not exists substrate_reduction_cursor (
    cursor_name text primary key,
    last_event_created_at timestamptz,
    last_event_id text,
    updated_at timestamptz not null default now()
);

create table if not exists substrate_node_biometrics_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists substrate_active_node_pressure_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists substrate_organ_emissions (
    emission_id text primary key,
    organ_id text not null,
    invocation_id text not null,
    emission_json jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists substrate_reduction_receipts (
    receipt_id text primary key,
    organ_id text,
    emission_id text,
    receipt_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_organ_emissions_created
  on substrate_organ_emissions (created_at desc);

create index if not exists idx_substrate_reduction_receipts_created
  on substrate_reduction_receipts (created_at desc);
