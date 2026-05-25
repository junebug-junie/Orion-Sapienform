create table if not exists substrate_tensor_slices (
    tensor_id text primary key,
    tensor_kind text not null,
    window_start timestamptz not null,
    window_end timestamptz not null,
    tensor_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_tensor_slices_kind
    on substrate_tensor_slices (tensor_kind);

create index if not exists idx_substrate_tensor_slices_window
    on substrate_tensor_slices (window_start, window_end);
