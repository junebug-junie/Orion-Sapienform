create table if not exists substrate_dispatch_results (
    result_id text primary key,
    dispatch_id text not null,
    frame_id text not null,
    status text not null,
    result_json jsonb not null,
    raw_len int not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_dispatch_results_dispatch_id
    on substrate_dispatch_results (dispatch_id);

create index if not exists idx_substrate_dispatch_results_created_at
    on substrate_dispatch_results (created_at desc);
