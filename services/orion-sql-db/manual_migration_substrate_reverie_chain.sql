-- Reverie chain store (Phase C). One row per completed train of thought
-- (ReverieChainV1). Apply before enabling ORION_REVERIE_CHAIN_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reverie_chain.sql

create table if not exists substrate_reverie_chain (
    chain_id text primary key,
    created_at timestamptz not null,
    theme_key text,
    terminal_reason text not null,
    ema_salience double precision not null default 0.0,
    committed_proposal_id text,
    chain_json jsonb not null,
    stored_at timestamptz not null default now()
);

create index if not exists idx_substrate_reverie_chain_created_at
    on substrate_reverie_chain (created_at desc);
