-- Compaction request queue (Phase E). Reverie (reasoning) drops a typed ask
-- here; the offline dream (storage) would drain it later. Applied by nothing
-- yet — this is a deliberate dead-end so we can watch WHAT reverie asks to
-- compact at zero risk. Apply before ORION_REVERIE_COMPACTION_REQUEST_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_dream_compaction_request_queue.sql

create table if not exists dream_compaction_request_queue (
    request_id text primary key,
    theme text not null,
    op_hint text not null default 'consolidate',
    reason text not null default '',
    origin_chain_id text,
    created_at timestamptz not null,
    request_json jsonb not null,
    consumed_at timestamptz,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_dream_compaction_request_queue_created_at
    on dream_compaction_request_queue (created_at desc);
