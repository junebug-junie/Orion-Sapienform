-- Append-only raw-event history for the codebase-mass domain (docs/superpowers/
-- specs/2026-07-30-codebase-mass-signal-design.md) -- one row per real
-- CodebaseDeltaV1 event actually consumed, unconditionally (not gated on
-- score > 0.0 the way substrate_reduction_receipts' audit trail is -- this
-- table's whole purpose is a complete real history for analytics, not a
-- curated "notable events only" log).
--
-- Deliberately separate from substrate_codebase_mass_baseline (that table
-- holds only the running EWMA state -- ewma/variance/n per sub-domain -- not
-- the raw per-tick payload: real git commit counts/line churn, real PR
-- numbers, real graph node/edge deltas). Without this table, that raw detail
-- is only ever visible transiently on the bus (ephemeral pub/sub, no
-- history) -- a "cocreation signals" analytics view showing anything beyond
-- the current aggregate state needs this real backing data first.
--
-- Same single-writer, brand-new-table shape as every other table in this
-- arc (substrate_codebase_mass_baseline, substrate_attention_self_model) --
-- this repo's own established fix for the "tick clobbers a field it doesn't
-- own" failure class.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_codebase_delta_log_v1.sql

create table if not exists substrate_codebase_delta_log (
    event_id text primary key,
    domain text not null,
    observed_at timestamptz not null,
    score numeric not null,
    event_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_codebase_delta_log_observed
    on substrate_codebase_delta_log(observed_at desc);

create index if not exists idx_codebase_delta_log_domain_observed
    on substrate_codebase_delta_log(domain, observed_at desc);
