create table if not exists substrate_execution_dispatch_frames (
    frame_id text primary key,
    source_policy_frame_id text not null,
    source_proposal_frame_id text not null,
    source_self_state_id text not null,
    generated_at timestamptz not null,
    policy_id text not null,
    dispatch_frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_execution_dispatch_frames_generated_at
    on substrate_execution_dispatch_frames (generated_at desc);

create index if not exists idx_substrate_execution_dispatch_frames_source_policy
    on substrate_execution_dispatch_frames (source_policy_frame_id);

create index if not exists idx_substrate_execution_dispatch_frames_source_self_state
    on substrate_execution_dispatch_frames (source_self_state_id);
