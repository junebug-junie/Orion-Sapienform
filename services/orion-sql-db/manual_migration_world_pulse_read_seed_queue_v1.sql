-- World-pulse Concept Atlas read pipeline seed queue v1
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_world_pulse_read_seed_queue_v1.sql

create table if not exists world_pulse_read_seed (
    seed_id text primary key,
    kind text not null check (kind in ('finding', 'digest_item')),
    run_id text not null,
    url text not null,
    title text not null default '',
    section text not null default '',
    item_id text null,
    priority int not null default 100,
    status text not null default 'pending'
        check (status in ('pending', 'claimed', 'done', 'failed', 'skipped')),
    trace_id text null,
    last_error text null,
    created_at timestamptz not null default now(),
    claimed_at timestamptz null,
    completed_at timestamptz null
);

create index if not exists idx_world_pulse_read_seed_claim
    on world_pulse_read_seed (status, priority, created_at);

create index if not exists idx_world_pulse_read_seed_run
    on world_pulse_read_seed (run_id);
