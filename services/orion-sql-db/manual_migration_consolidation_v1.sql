create table if not exists substrate_consolidation_frames (
    frame_id text primary key,
    window_start timestamptz not null,
    window_end timestamptz not null,
    generated_at timestamptz not null,
    policy_id text not null,
    consolidation_frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_consolidation_frames_generated_at
    on substrate_consolidation_frames (generated_at desc);

create index if not exists idx_substrate_consolidation_frames_window
    on substrate_consolidation_frames (window_start, window_end);
