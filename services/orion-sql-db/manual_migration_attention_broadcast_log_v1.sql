-- Append-only companion to substrate_attention_broadcast_projection (rung 3
-- GWT-dispatch/Lamme lane). The projection table is a singleton upsert
-- (PRIMARY KEY on projection_id, exactly one row, overwritten every ~30s
-- tick) with no per-tick history -- confirmed live while building
-- scripts/analysis/measure_ast_hot_reducer.py (2026-07-18), which could only
-- ever join the single current snapshot to the tail of ticks at or after its
-- own timestamp and could never demonstrate a real historical
-- voluntary_override event. This table appends one row per broadcast tick so
-- that replay can search real history instead. Apply before relying on
-- historical broadcast-lane replay; the existing singleton table, its writer
-- (save_attention_broadcast), and AttentionBroadcastProjectionV1 are
-- untouched -- this is purely additive.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_broadcast_log_v1.sql

create table if not exists substrate_attention_broadcast_log (
    log_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);
create index if not exists idx_attention_broadcast_log_generated on substrate_attention_broadcast_log(generated_at desc);
