-- Reverie semantic lift v1: human referents for harness_closure pointers.
-- Apply before enabling ORION_REVERIE_SEMANTIC_LIFT_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql

create table if not exists substrate_turn_referent (
    correlation_id          text primary key,
    coalition_ref           text not null,
    user_message_excerpt    text not null default '',
    stance_imperative       text not null default '',
    surprise_unresolved     boolean not null default true,
    thought_event_id        text,
    created_at              timestamptz not null,
    stored_at               timestamptz not null default now()
);

create index if not exists idx_substrate_turn_referent_created_at
    on substrate_turn_referent (created_at desc);

create index if not exists idx_substrate_turn_referent_coalition_ref
    on substrate_turn_referent (coalition_ref);
