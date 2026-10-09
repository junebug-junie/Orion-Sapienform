-- Manual, after manual_migration_durable_resource_admission_v1.sql (tests/evals apply it
-- via PostgresAdmissionStore.setup). Additive and idempotent; optional for correctness.
-- reconcile re-reads unconfirmed reverie.visual abandons (event 'run.abandon_pending')
-- every hold-status poll; without this the lookup scans the whole event log.
CREATE INDEX IF NOT EXISTS durable_resource_abandon_pending
    ON durable_resource_events (generated_at, run_id) WHERE event = 'run.abandon_pending';
