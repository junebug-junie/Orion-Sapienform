create table if not exists substrate_schema_candidates (
    schema_candidate_id text primary key,
    candidate_kind text not null,
    label text not null,
    promotion_status text not null,
    schema_candidate_json jsonb not null,
    updated_at timestamptz not null default now()
);

create index if not exists idx_substrate_schema_candidates_kind
    on substrate_schema_candidates (candidate_kind);

create index if not exists idx_substrate_schema_candidates_status
    on substrate_schema_candidates (promotion_status);
