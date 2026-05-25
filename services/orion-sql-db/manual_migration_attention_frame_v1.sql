create table if not exists substrate_attention_frames (
    frame_id text primary key,
    source_field_tick_id text not null,
    source_field_generated_at timestamptz not null,
    generated_at timestamptz not null,
    policy_id text not null,
    frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_attention_frames_generated_at
    on substrate_attention_frames (generated_at desc);

create index if not exists idx_substrate_attention_frames_source_tick
    on substrate_attention_frames (source_field_tick_id);
