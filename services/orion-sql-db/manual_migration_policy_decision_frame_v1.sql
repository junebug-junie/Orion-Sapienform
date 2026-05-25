create table if not exists substrate_policy_decision_frames (
    frame_id text primary key,
    source_proposal_frame_id text not null,
    source_self_state_id text not null,
    generated_at timestamptz not null,
    policy_id text not null,
    policy_decision_frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_policy_decision_frames_generated_at
    on substrate_policy_decision_frames (generated_at desc);

create index if not exists idx_substrate_policy_decision_frames_source_proposal
    on substrate_policy_decision_frames (source_proposal_frame_id);

create index if not exists idx_substrate_policy_decision_frames_source_self_state
    on substrate_policy_decision_frames (source_self_state_id);
