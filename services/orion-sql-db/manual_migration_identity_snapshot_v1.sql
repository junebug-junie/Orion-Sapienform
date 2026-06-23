-- Identity snapshot v1 (apply before orion-self-state-runtime)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_identity_snapshot_v1.sql

create table if not exists identity_snapshots (
    snapshot_id text primary key,
    source_self_state_id text not null,
    generated_at timestamptz not null,
    dominant_drive text not null,
    self_state_condition text not null,
    snapshot_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_identity_snapshots_generated_at
    on identity_snapshots (generated_at desc);

create index if not exists idx_identity_snapshots_source_self_state
    on identity_snapshots (source_self_state_id);
