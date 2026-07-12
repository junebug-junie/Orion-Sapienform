-- Field digester v1 (apply before orion-field-digester)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_field_digester_v1.sql

create table if not exists substrate_field_digest_cursor (
    cursor_name text primary key,
    last_receipt_created_at timestamptz,
    last_receipt_id text,
    updated_at timestamptz not null default now()
);

create table if not exists substrate_field_applied_deltas (
    delta_id text primary key,
    receipt_id text not null,
    applied_at timestamptz not null default now()
);

create index if not exists idx_substrate_field_applied_deltas_receipt
  on substrate_field_applied_deltas (receipt_id);

-- Backs prune_applied_deltas' "WHERE applied_at < :cutoff ORDER BY applied_at ASC"
-- scan; without it, every prune batch does a full sequential scan + sort of this
-- table. On an already-live database, apply separately with CONCURRENTLY instead
-- (see services/orion-field-digester/README.md) to avoid a write lock.
create index if not exists idx_substrate_field_applied_deltas_applied_at
  on substrate_field_applied_deltas (applied_at);

create table if not exists substrate_field_state (
    tick_id text primary key,
    generated_at timestamptz not null,
    field_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_field_state_generated
  on substrate_field_state (generated_at desc);
