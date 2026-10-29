-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Per-call token ledger (2026-10-29). One row per LLM call that reached a model.
--
-- Writer: orion-sql-writer off bus orion:llm:call:usage (LlmCallUsageV1, orion/schemas/llm_call_usage.py).
-- Producers: orion-llm-gateway (recorder = 'gateway': bus RPC + /v1/chat/completions + /v1/messages)
-- and orion-harness-governor's warm FCC relay (recorder = 'harness_relay': FCC-path turns only, whose
-- gateway row has no correlation_id).
--
-- Join rules (no double count):
--   totals over time (tokens/hour, joules/token): WHERE recorder = 'gateway'
--   one turn or act:                              WHERE correlation_id = <corr>
-- prompt_tokens INCLUDES cached_prompt_tokens (prompt tokens reused from the KV cache).
-- duration_ms: gateway = grant -> reply (model time, queue excluded); harness_relay = request -> reply end.
-- sql-writer's create_all makes the same table at boot; this file is the operator contract.
-- Additive only. Insert-only, PK call_id.

CREATE TABLE IF NOT EXISTS llm_call_usage (
    call_id               text PRIMARY KEY,
    recorded_at           timestamptz NOT NULL,
    recorder              text NOT NULL,
    path                  text NOT NULL,
    correlation_id        text,
    route                 text,
    model                 text,
    served_by             text,
    pool_role             text,
    holder                text,
    verb                  text,
    node                  text,
    outcome               text NOT NULL,
    stream                boolean NOT NULL,
    prompt_tokens         integer,
    cached_prompt_tokens  integer,
    completion_tokens     integer,
    token_source          text NOT NULL,
    duration_ms           integer,
    wait_ms               integer,
    decode_tps            double precision,
    created_at            timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_call_usage_recorded ON llm_call_usage (recorded_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_usage_corr_recorded ON llm_call_usage (correlation_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_usage_route_recorded ON llm_call_usage (route, recorded_at);
