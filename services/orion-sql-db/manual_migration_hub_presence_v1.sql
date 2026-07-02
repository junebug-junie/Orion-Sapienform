-- Hub presence v1 (self-observability: Orion's chat liveness, single-row upsert)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_hub_presence_v1.sql

create table if not exists substrate_hub_presence (
    presence_id text primary key,
    generated_at timestamptz not null,
    presence_json jsonb not null,
    updated_at timestamptz not null default now()
);
