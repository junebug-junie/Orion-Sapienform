-- ORION-MIGRATION-SUPERSEDED-BY: manual_migration_proposal_frame_v2_drop_self_state.sql
-- That migration explicitly drops idx_*_source_self_state and the NOT NULL on
-- source_self_state_id, because nothing populates the column going forward. This file
-- is correct history; its index is correctly ABSENT from the live database.
-- Marker read by scripts/check_sql_migrations_applied.py.

create table if not exists substrate_proposal_frames (
    frame_id text primary key,
    source_self_state_id text not null,
    source_attention_frame_id text,
    source_field_tick_id text,
    generated_at timestamptz not null,
    policy_id text not null,
    proposal_frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_proposal_frames_generated_at
    on substrate_proposal_frames (generated_at desc);

create index if not exists idx_substrate_proposal_frames_source_self_state
    on substrate_proposal_frames (source_self_state_id);
