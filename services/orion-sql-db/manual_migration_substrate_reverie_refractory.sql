-- Reverie refractory table (Phase C). A resolved theme is suppressed as a chain
-- trigger until suppressed_until, so a discharged loop cannot immediately
-- re-ignite (ouroboros habituation). Apply before ORION_REVERIE_CHAIN_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reverie_refractory.sql

create table if not exists substrate_reverie_refractory (
    theme_key text primary key,
    suppressed_until timestamptz not null,
    updated_at timestamptz not null default now()
);

create index if not exists idx_substrate_reverie_refractory_until
    on substrate_reverie_refractory (suppressed_until);
