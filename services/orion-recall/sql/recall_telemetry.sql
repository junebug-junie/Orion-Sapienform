-- Idempotent table for recall telemetry persistence
CREATE TABLE IF NOT EXISTS recall_telemetry (
    id uuid PRIMARY KEY,
    corr_id text,
    session_id text NULL,
    node_id text NULL,
    verb text NULL,
    profile text,
    query text,
    selected_ids jsonb,
    backend_counts jsonb,
    latency_ms integer,
    created_at timestamptz DEFAULT now()
);

-- Bounded-retrieval telemetry (2026-09-29). Nullable, additive. The service
-- also applies these in _persist_decision's once-per-process DDL.
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS query_chars integer;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS retrieval_query_source text;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS sub_query_count integer;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS candidates_fetched integer;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS candidates_kept integer;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS deadline_hit boolean;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS timings_ms jsonb;
