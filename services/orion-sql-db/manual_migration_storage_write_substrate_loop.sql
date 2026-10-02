-- storage_write substrate projection (apply BEFORE ENABLE_STORAGE_WRITE_REDUCER=true on substrate-runtime)
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_storage_write_substrate_loop.sql
--
-- This migration:
--   1. Creates substrate_storage_write_projection (one row, projection_id
--      active_storage_write_projection: the sql-writer's latest window, its
--      rolling 600 s span, and the write_failure_pressure reading).
--   2. Seeds the storage_write_grammar_reducer cursor row (null position; the
--      runtime tail-seeds it on first poll, so no history is replayed).
-- Idempotent.

create table if not exists substrate_storage_write_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

insert into substrate_reduction_cursor (cursor_name, last_event_created_at, last_event_id, updated_at)
values ('storage_write_grammar_reducer', null, null, now())
on conflict (cursor_name) do nothing;
