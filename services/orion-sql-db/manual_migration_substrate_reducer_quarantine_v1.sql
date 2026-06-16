-- Durable reducer poison quarantine + operator acknowledgement audit trail.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reducer_quarantine_v1.sql

create table if not exists substrate_reducer_quarantine (
    quarantine_id text primary key,
    reducer_key text not null,
    cursor_name text not null,
    event_id text not null,
    trace_id text,
    reason text not null,
    quarantined_at timestamptz not null default now(),
    acknowledged_at timestamptz,
    acknowledged_by text
);

create unique index if not exists idx_substrate_quarantine_reducer_event
  on substrate_reducer_quarantine (reducer_key, event_id);

create index if not exists idx_substrate_quarantine_unacked
  on substrate_reducer_quarantine (cursor_name, quarantined_at desc)
  where acknowledged_at is null;

create index if not exists idx_substrate_quarantine_quarantined_at
  on substrate_reducer_quarantine (quarantined_at desc);
