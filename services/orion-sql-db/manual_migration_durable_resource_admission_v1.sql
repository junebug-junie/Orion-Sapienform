-- Apply before enabling DURABLE_RESOURCE_ADMISSION_ENABLED.
-- LangGraph owns checkpoint DDL and workflow state. These are inbox/resource facts.
CREATE TABLE IF NOT EXISTS durable_admission_runs (
    run_id text PRIMARY KEY,
    request jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    control text CHECK (control IN ('paused', 'cancelled')),
    terminal text,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS durable_resource_demands (
    demand_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES durable_admission_runs(run_id),
    requirement jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    status text NOT NULL CHECK (status IN ('pending', 'suspended', 'granted', 'withdrawn')),
    decision jsonb NOT NULL DEFAULT '{}',
    UNIQUE (run_id)
);
CREATE INDEX IF NOT EXISTS durable_resource_demands_fifo
    ON durable_resource_demands (created_at, demand_id) WHERE status = 'pending';
CREATE SEQUENCE IF NOT EXISTS durable_resource_fencing_generation;
CREATE TABLE IF NOT EXISTS durable_resource_leases (
    lease_id text PRIMARY KEY,
    demand_id text NOT NULL REFERENCES durable_resource_demands(demand_id),
    run_id text NOT NULL REFERENCES durable_admission_runs(run_id),
    resource_key text NOT NULL,
    lane text NOT NULL,
    backend_key text NOT NULL,
    generation bigint NOT NULL UNIQUE DEFAULT nextval('durable_resource_fencing_generation'),
    granted_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    heartbeat_at timestamptz NOT NULL,
    status text NOT NULL CHECK (status IN ('active', 'released', 'expired')),
    CHECK (expires_at > granted_at)
);
CREATE UNIQUE INDEX IF NOT EXISTS durable_resource_one_active_run
    ON durable_resource_leases (run_id) WHERE status = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS durable_resource_one_active_lane
    ON durable_resource_leases (resource_key) WHERE status = 'active';
-- Route aliases resolving to one actual backend never manufacture capacity.
CREATE UNIQUE INDEX IF NOT EXISTS durable_resource_one_active_backend
    ON durable_resource_leases (backend_key) WHERE status = 'active';
CREATE TABLE IF NOT EXISTS durable_resource_events (
    entry_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES durable_admission_runs(run_id),
    event text NOT NULL,
    generated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    payload jsonb NOT NULL,
    published_at timestamptz
);
CREATE INDEX IF NOT EXISTS durable_resource_history
    ON durable_resource_events (run_id, generated_at, entry_id);
CREATE INDEX IF NOT EXISTS durable_resource_outbox
    ON durable_resource_events (generated_at, entry_id) WHERE published_at IS NULL;
