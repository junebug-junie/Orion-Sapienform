-- Transport bus substrate projection (apply before enabling transport reducer)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_transport_substrate_loop.sql

create table if not exists substrate_transport_bus_projection (
    projection_id text primary key,
    projection_json jsonb not null,
    updated_at timestamptz not null default now()
);

create table if not exists substrate_transport_bus_cursor (
    cursor_id text primary key,
    last_created_at timestamptz,
    last_event_id text,
    updated_at timestamptz not null default now()
);
