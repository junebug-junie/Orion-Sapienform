create table if not exists substrate_feedback_frames (
    frame_id text primary key,
    source_execution_dispatch_frame_id text not null,
    source_policy_frame_id text,
    source_proposal_frame_id text,
    source_self_state_id text,
    generated_at timestamptz not null,
    policy_id text not null,
    feedback_frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_feedback_frames_generated_at
    on substrate_feedback_frames (generated_at desc);

create index if not exists idx_substrate_feedback_frames_source_dispatch
    on substrate_feedback_frames (source_execution_dispatch_frame_id);
