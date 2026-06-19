-- Chat grammar substrate projection (apply before enabling ENABLE_CHAT_GRAMMAR_REDUCER=true on substrate-runtime)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_chat_substrate_loop.sql
--
-- This migration:
--   1. Creates the substrate_chat_session_projection table for chat grammar lane projections.
--   2. Adds an index on generated_at for efficient latest-projection lookups.
--   3. Seeds a cursor row in substrate_reduction_cursor for the chat_grammar_consumer lane.

create table if not exists substrate_chat_session_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_chat_proj_generated_at
    on substrate_chat_session_projection (generated_at desc);

insert into substrate_reduction_cursor (cursor_name, last_event_created_at, last_event_id, updated_at)
values ('chat_grammar_consumer', null, null, now())
on conflict (cursor_name) do nothing;
