-- Compaction delta (Phase F). REM narration proposes what sleep *would* do to
-- memory (consolidate / downscale / prune). This is a STAGED table: the delta is
-- persisted and previewed in the hub, but applied by nothing — the applier
-- (Phase G) is a separate, hard-gated process (ORION_DREAM_COMPACTION_APPLY_ENABLED).
-- Every row here is proposal_marked=true. Apply before ORION_DREAM_REM_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_dream_compaction_delta.sql

create table if not exists dream_compaction_delta (
    delta_id text primary key,
    dream_id text,
    created_at timestamptz not null,
    cards_out integer not null default 0,
    edges_downscaled integer not null default 0,
    rows_pruned integer not null default 0,
    bytes_reclaimed_est bigint not null default 0,
    -- Hard invariant mirrored from the schema: Phase-F deltas are proposals only.
    proposal_marked boolean not null default true,
    -- Set only if/when a Phase-G applier consumes it (stays null in Phase F).
    applied_at timestamptz,
    delta_json jsonb not null,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_dream_compaction_delta_created_at
    on dream_compaction_delta (created_at desc);
