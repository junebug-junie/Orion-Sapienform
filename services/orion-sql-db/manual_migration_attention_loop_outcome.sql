-- Human close-the-loop labels (Resolve/Dismiss) + implicit decayed_unattended.
-- The sparse-but-clean label table for the salience refit. Written directly by
-- orion-hub when Juniper acts. Apply before ORION_ATTENTION_PENDING_CARDS_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_loop_outcome.sql

create table if not exists attention_loop_outcome (
    outcome_id text primary key,
    loop_id text not null,
    theme_key text not null,
    verdict text not null,
    actor text not null default 'juniper',
    note text not null default '',
    salience_at_close double precision not null default 0,
    weights_version text not null default 'seed-v1',
    features_at_close jsonb not null default '{}'::jsonb,
    created_at timestamptz not null,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_attention_loop_outcome_created_at
    on attention_loop_outcome (created_at desc);

create index if not exists idx_attention_loop_outcome_verdict
    on attention_loop_outcome (verdict, created_at desc);
