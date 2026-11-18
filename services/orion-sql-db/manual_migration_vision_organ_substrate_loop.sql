-- ORION-MIGRATION-REQUIRED-BY: orion-substrate-runtime
-- vision_organ substrate projection (apply before ENABLE_VISION_ORGAN_REDUCER=true on substrate-runtime)
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_organ_substrate_loop.sql
--
-- This migration:
--   1. Creates substrate_vision_organ_projection (one row, projection_id
--      active_vision_organ_projection: latest router window per camera stream,
--      the organ readings, and the rolling task-outcome window).
--   2. Seeds the vision_organ_grammar_reducer cursor row (null position; the
--      runtime tail-seeds it on first poll, so no history is replayed).
-- Idempotent.

create table if not exists substrate_vision_organ_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

insert into substrate_reduction_cursor (cursor_name, last_event_created_at, last_event_id, updated_at)
values ('vision_organ_grammar_reducer', null, null, now())
on conflict (cursor_name) do nothing;
