-- causal_geometry_snapshots: Causal Geometry v1 Phase A measurement persistence.
-- Producer: orion-field-digester (orion/substrate/causal_geometry_snapshot_store.py,
-- called from orion/substrate/causal_geometry_producer.py's scheduled production
-- cycle, off by default via FIELD_PLASTICITY_PRODUCER_ENABLED). Writes one row per
-- successful measurement cycle -- never via the bus, never via orion-sql-writer
-- (see orion/bus/channels.yaml's orion:causal_geometry:snapshot entry, kept
-- registered for schema validity but with producer_services: [] -- routing this
-- through sql-writer's shared worker was judged too invasive for the value).
--
-- Consumer: orion-hub (services/orion-hub/scripts/api_routes.py's
-- /api/causal-geometry/snapshot and /api/causal-geometry/history endpoints), read
-- via the hub's existing memory_pg_pool (asyncpg), the same pool already used for
-- memory cards against this same conjourney database.
--
-- Retention: pruned by orion-field-digester's own _prune_tick() (same loop as its
-- other tables), governed by FIELD_PLASTICITY_SNAPSHOT_RETENTION_HOURS (default
-- 720h / 30 days), always keeping at least the single most-recent row.
--
-- On boot, orion-field-digester applies this same DDL lazily on first write via
-- orion/substrate/causal_geometry_snapshot_store.py's ensure_schema() (CREATE TABLE/
-- INDEX IF NOT EXISTS, idempotent). This file is the standalone equivalent for
-- manual application against a running Postgres; it is a harmless no-op once
-- applied, or once the producer has run at least once.

CREATE TABLE IF NOT EXISTS causal_geometry_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    designed_topology_version TEXT,
    insufficient_data BOOLEAN NOT NULL,
    edges_json JSONB NOT NULL,
    divergence_json JSONB NOT NULL,
    notes_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_causal_geometry_snapshots_generated_at
    ON causal_geometry_snapshots (generated_at DESC);
