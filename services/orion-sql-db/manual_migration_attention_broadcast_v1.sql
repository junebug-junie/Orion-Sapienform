-- Continuous attention broadcast (self-modeling loop rung 3): single-row
-- projection of the current substrate workspace winner. Apply before enabling
-- ORION_ATTENTION_BROADCAST_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_broadcast_v1.sql

create table if not exists substrate_attention_broadcast_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);
