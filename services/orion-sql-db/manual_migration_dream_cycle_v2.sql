-- Dream cycle v2: sleep pressure -> replay -> REM recombination -> hypotheses.
-- orion-dream writes all three tables directly (same pattern as
-- dream_compaction_delta). Nothing here touches canonical memory.
--
-- ONE CROSS-SERVICE COLUMN PAIR: orion-hub's curiosity loop stamps
-- dream_hypothesis.offered_at / offered_run_id when it shows a hypothesis to
-- Orion (orion/dream/hypotheses.py::TAKE_FOR_OFFER_SQL). Hub writes nothing
-- else here. `arm` is never read by Hub.
--
-- Apply before ORION_DREAM_CYCLE_ENABLED / HUB_CURIOSITY_DREAM_HYPOTHESES_ENABLED:
--   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_dream_cycle_v2.sql
-- Rollback (drops only v2 state):
--   drop table if exists dream_hypothesis, dream_replay_item, dream_cycle;

create table if not exists dream_cycle (
    cycle_id text primary key,
    trigger text not null,
    status text not null,
    started_at timestamptz not null,
    ended_at timestamptz not null,
    pressure double precision not null default 0,
    replay_count integer not null default 0,
    hypothesis_count integer not null default 0,
    no_link_count integer not null default 0,
    llm_failures integer not null default 0,
    compaction_delta_id text,
    cycle_json jsonb not null,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_dream_cycle_ended_at on dream_cycle (ended_at desc);

create table if not exists dream_replay_item (
    cycle_id text not null references dream_cycle(cycle_id) on delete cascade,
    ref_id text not null,
    rank integer not null,
    source_kind text not null,
    weight double precision not null,
    reason text not null,
    primary key (cycle_id, ref_id)
);

create table if not exists dream_hypothesis (
    hypothesis_id text primary key,
    cycle_id text not null references dream_cycle(cycle_id) on delete cascade,
    arm text not null check (arm in ('dream', 'control')),
    claim text not null,
    why text not null default '',
    ref_a text not null,
    ref_b text not null,
    created_at timestamptz not null,
    expires_at timestamptz not null,
    offered_at timestamptz,
    offered_run_id text
);

create index if not exists idx_dream_hypothesis_offerable
    on dream_hypothesis (expires_at) where offered_at is null;
