-- Self-sense eval log v1 (Patch A of docs/superpowers/specs/2026-09-08-orion-sense-of-self-design.md)
-- One row per fixed question per run of services/orion-hub/evals/run_self_sense_eval.py.
-- orion-sql-writer also creates this table at boot via Base.metadata.create_all
-- (same as self_concept_history); this file is the explicit, idempotent copy for
-- an operator applying it ahead of a sql-writer redeploy.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_self_sense_eval_log_v1.sql

create table if not exists self_sense_eval_log (
    entry_id text primary key,
    run_id text not null,
    created_at timestamptz not null default now(),
    question_key text not null,
    question text not null,
    answer_text text not null,
    answer_source text not null,
    correlation_id text,
    self_label_score integer not null,
    grounded_record_score integer not null,
    self_definition_version integer,
    notes text
);

create index if not exists idx_self_sense_eval_log_created_at
    on self_sense_eval_log (created_at);

create index if not exists idx_self_sense_eval_log_run_id
    on self_sense_eval_log (run_id);
