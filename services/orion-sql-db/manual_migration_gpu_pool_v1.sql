-- orion-gpu-pool projection (spec: docs/superpowers/specs/2026-09-24-gpu-pool-design.md).
-- Apply before starting orion-gpu-pool; the service refuses to boot without these tables.
-- The lease history lives in LangGraph checkpoints (created by the saver itself) and in
-- gpu_pool_events (written by orion-sql-writer from orion:gpu_pool:event).
CREATE TABLE IF NOT EXISTS gpu_pool_leases (
    lease_id text PRIMARY KEY,
    request_id text NOT NULL UNIQUE,
    holder text NOT NULL,
    work_class text NOT NULL,
    priority text NOT NULL,
    kind text NOT NULL,
    status text NOT NULL,
    role text,
    attempt integer NOT NULL DEFAULT 1,
    generation integer NOT NULL DEFAULT 0,
    operator boolean NOT NULL DEFAULT false,
    min_ctx_tokens integer NOT NULL DEFAULT 0,
    needs_vision boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL,
    queued_since timestamptz,
    granted_at timestamptz,
    recall_by timestamptz,
    not_before timestamptz,
    deadline_at timestamptz,
    expires_at timestamptz,
    turn_correlation_id text,
    parent_lease_id text,
    reason text,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS gpu_pool_leases_live_idx ON gpu_pool_leases (status)
    WHERE status NOT IN ('released');
CREATE INDEX IF NOT EXISTS gpu_pool_leases_class_created_idx ON gpu_pool_leases (work_class, created_at);
CREATE INDEX IF NOT EXISTS gpu_pool_leases_holder_idx ON gpu_pool_leases (holder, created_at);

CREATE TABLE IF NOT EXISTS gpu_pool_cards (
    card text PRIMARY KEY,
    lent boolean NOT NULL DEFAULT false,
    swapped_in text[] NOT NULL DEFAULT '{}',
    swap_state text NOT NULL DEFAULT 'idle',
    cooldown_until timestamptz,
    last_active_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    updated_by text
);
