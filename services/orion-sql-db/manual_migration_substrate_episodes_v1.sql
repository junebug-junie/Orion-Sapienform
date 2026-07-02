-- Episodic continuity (self-modeling loop rung 4): proposal-marked rollups of
-- reduction-receipt windows. Apply before enabling SUBSTRATE_EPISODIC_TICK_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_episodes_v1.sql

create table if not exists substrate_episode_summaries (
    episode_id text primary key,
    status text not null default 'proposal',
    window_start timestamptz not null,
    window_end timestamptz not null,
    episode_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_episode_summaries_window_end
    on substrate_episode_summaries (window_end desc);
