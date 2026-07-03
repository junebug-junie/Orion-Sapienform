-- Endogenous curiosity candidates v1 (apply before enabling the felt-state curiosity lane)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_curiosity_candidates_v1.sql

create table if not exists substrate_endogenous_curiosity_candidates (
    candidate_set_id text primary key,
    generated_at timestamptz not null,
    candidates_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_endogenous_curiosity_generated_at
    on substrate_endogenous_curiosity_candidates (generated_at desc);
