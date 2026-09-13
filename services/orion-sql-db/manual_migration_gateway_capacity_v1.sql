-- Apply after manual_migration_durable_resource_admission_v1.sql, before
-- DURABLE_RUNS_CAPACITY_ENABLED=true. No checkpoint/workflow schema changes.
CREATE TABLE IF NOT EXISTS durable_gateway_permits (
    request_id text PRIMARY KEY,
    permit_id text NOT NULL UNIQUE,
    request jsonb NOT NULL,
    correlation_id text NOT NULL,
    lane text NOT NULL,
    backend_key text NOT NULL,
    lease_id text REFERENCES durable_resource_leases(lease_id),
    generation bigint,
    max_inflight integer NOT NULL CHECK (max_inflight BETWEEN 1 AND 128),
    deadline_at timestamptz NOT NULL,
    granted_at timestamptz NOT NULL,
    heartbeat_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    status text NOT NULL CHECK (status IN ('active', 'released', 'expired')),
    CHECK (expires_at > granted_at),
    CHECK ((lease_id IS NULL) = (generation IS NULL))
);
CREATE INDEX IF NOT EXISTS durable_gateway_active_backend
    ON durable_gateway_permits (backend_key, expires_at) WHERE status='active';
CREATE UNIQUE INDEX IF NOT EXISTS durable_gateway_one_owner_call
    ON durable_gateway_permits (lease_id) WHERE status='active' AND lease_id IS NOT NULL;
