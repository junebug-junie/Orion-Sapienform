-- Resonance alerts (Phase H). The automated ouroboros tripwire persists a row
-- whenever a theme re-ignites faster than its refractory bound allows. Observation
-- only — nothing here mutates cognition; it is the guard that licenses turning the
-- compaction applier's hot gate on. Apply before ORION_REVERIE_RESONANCE_ALERT_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reverie_resonance_alert.sql

create table if not exists substrate_reverie_resonance_alert (
    alert_id text primary key,
    theme_key text not null,
    violation_count integer not null default 0,
    refractory_sec double precision not null default 0,
    min_gap_sec double precision not null default 0,
    occurrences integer not null default 0,
    created_at timestamptz not null,
    alert_json jsonb not null,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_substrate_reverie_resonance_alert_created_at
    on substrate_reverie_resonance_alert (created_at desc);

create index if not exists idx_substrate_reverie_resonance_alert_theme
    on substrate_reverie_resonance_alert (theme_key, created_at desc);
