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
