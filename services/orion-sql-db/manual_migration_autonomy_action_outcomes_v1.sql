-- Autonomy action outcomes v1 — closes the goal action → autonomy reducer feedback loop
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_autonomy_action_outcomes_v1.sql

create table if not exists autonomy_action_outcomes (
    outcome_id text primary key,
    action_id text not null,
    kind text not null,
    success boolean,
    surprise float not null default 0.0,
    observed_at timestamptz,
    outcome_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_autonomy_action_outcomes_created_at
    on autonomy_action_outcomes (created_at desc);

create index if not exists idx_autonomy_action_outcomes_action_id
    on autonomy_action_outcomes (action_id);
