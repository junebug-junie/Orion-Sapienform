-- Computed salience telemetry. One row per scored loop (feature vector + score).
-- The input half of the learning join; refit_salience_weights.py reads it with
-- attention_loop_outcome. Written directly by orion-thought (best-effort), same
-- pattern as substrate_reverie_thought. Observation only — mutates no cognition.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_salience_trace.sql

create table if not exists attention_salience_trace (
    trace_id text primary key,
    loop_id text not null,
    theme_key text not null,
    description text not null default '',
    correlation_id text,
    salience double precision not null default 0,
    weights_version text not null default 'seed-v1',
    scope text not null default 'reverie',
    features jsonb not null default '{}'::jsonb,
    created_at timestamptz not null,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_attention_salience_trace_created_at
    on attention_salience_trace (created_at desc);

create index if not exists idx_attention_salience_trace_theme
    on attention_salience_trace (theme_key, created_at desc);
