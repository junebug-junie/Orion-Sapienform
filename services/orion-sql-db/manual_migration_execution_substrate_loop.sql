-- Execution substrate trajectory projection (apply before enabling execution reducer)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_execution_substrate_loop.sql

create table if not exists substrate_execution_trajectory_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);
