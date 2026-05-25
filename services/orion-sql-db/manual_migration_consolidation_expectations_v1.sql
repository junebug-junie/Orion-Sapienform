create table if not exists substrate_expectations (
    expectation_id text primary key,
    trigger_motif_id text not null,
    expected_outcome_kind text not null,
    expectation_json jsonb not null,
    updated_at timestamptz not null default now()
);

create index if not exists idx_substrate_expectations_trigger_motif
    on substrate_expectations (trigger_motif_id);

create index if not exists idx_substrate_expectations_outcome
    on substrate_expectations (expected_outcome_kind);
