-- Reverie spontaneous-thought store (Phase A of the reverie/dream weave).
-- Backs the hub _reverie_section panel: recent SpontaneousThoughtV1 emissions
-- narrating the current winning coalition (rung 3). Apply before enabling
-- ORION_REVERIE_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reverie_thought.sql

create table if not exists substrate_reverie_thought (
    thought_id text primary key,
    correlation_id text not null,
    created_at timestamptz not null,
    salience double precision not null default 0.0,
    interpretation text not null default '',
    thought_json jsonb not null,
    stored_at timestamptz not null default now()
);

create index if not exists idx_substrate_reverie_thought_created_at
    on substrate_reverie_thought (created_at desc);
