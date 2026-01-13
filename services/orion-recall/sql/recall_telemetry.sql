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
