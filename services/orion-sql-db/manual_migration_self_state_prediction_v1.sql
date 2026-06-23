-- Self-state prediction v1 (apply after manual_migration_self_state_v1.sql)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_self_state_prediction_v1.sql

create table if not exists self_state_predictions (
    prediction_id text primary key,
    source_self_state_id text not null,
    generated_at timestamptz not null,
    prediction_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_self_state_predictions_generated_at
    on self_state_predictions (generated_at desc);

create index if not exists idx_self_state_predictions_source
    on self_state_predictions (source_self_state_id);
