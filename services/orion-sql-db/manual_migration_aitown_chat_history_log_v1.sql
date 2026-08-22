-- AI Town chat-history table split, Phase 1 (docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_aitown_chat_history_log_v1.sql
--
-- Mirror of chat_history_log, column for column, populated by
-- services/orion-sql-writer/app/worker.py's dual-write path
-- (SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED) for rows whose client_meta carries
-- the AI Town platform tag. Additive only: chat_history_log itself is
-- completely untouched by this migration or by Phase 1's dual-write --
-- every existing reader of chat_history_log keeps working exactly as
-- before. Nothing reads this table yet; that is Phase 2's job.

-- Column types mirror the real live chat_history_log schema exactly
-- (checked via \d chat_history_log, 2026-08-19) -- character varying, not
-- text, for the short-string columns.
create table if not exists aitown_chat_history_log (
    id character varying primary key,
    correlation_id character varying,
    source character varying,
    prompt text,
    response text,
    user_id character varying,
    session_id character varying,
    spark_meta jsonb,
    memory_status character varying,
    memory_tier character varying,
    memory_reason character varying,
    thought_process text,
    client_meta jsonb,
    llm_uncertainty_source character varying,
    llm_mean_logprob double precision,
    llm_min_logprob double precision,
    llm_mean_top1_margin double precision,
    llm_low_margin_token_count integer,
    llm_low_logprob_token_count integer,
    llm_unstable_span_count integer,
    llm_uncertainty_available boolean,
    created_at timestamp without time zone default now()
);

create index if not exists idx_aitown_chat_history_log_correlation_id
    on aitown_chat_history_log (correlation_id);

create index if not exists idx_aitown_chat_history_log_memory_status
    on aitown_chat_history_log (memory_status);

create index if not exists idx_aitown_chat_history_log_memory_tier
    on aitown_chat_history_log (memory_tier);

-- Added 2026-08-19 (code review on the Phase-2 recall migration): every
-- orion-recall chat-content query filters and/or ORDER BYs on created_at
-- against this table (fetch_recent_fragments, fetch_related_by_entities,
-- fetch_exact_fragments, fetch_chat_turn_timestamps, fetch_chat_history_pairs,
-- fetch_sql_fragments). Without this, those queries force a sequential scan
-- on this table's side of every UNION ALL/separate query as row count grows.
-- Note: chat_history_log itself has the SAME pre-existing gap (no
-- created_at index at all, confirmed live via pg_indexes) -- not
-- introduced by this migration, not fixed here (out of scope: that table
-- is not owned by this migration file), but worth knowing.
create index if not exists idx_aitown_chat_history_log_created_at
    on aitown_chat_history_log (created_at);
