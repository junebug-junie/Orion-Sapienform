-- causal_geometry_snapshots: Causal Geometry v1 Phase A measurement persistence.
-- Producer: orion-field-digester (orion/substrate/causal_geometry_bus_publish.py,
-- called from orion/substrate/causal_geometry_producer.py's scheduled production
-- cycle, off by default via FIELD_PLASTICITY_PRODUCER_ENABLED) publishes
-- CausalGeometrySnapshotV1 on the bus channel orion:causal_geometry:snapshot
-- (kind causal.geometry.snapshot.v1) -- one message per successful measurement
-- cycle. orion-sql-writer consumes it via its standard MODEL_MAP/DEFAULT_ROUTE_MAP
-- routing (services/orion-sql-writer/app/models/causal_geometry_snapshot.py,
-- CausalGeometrySnapshotSQL) and writes this table. Column names deliberately
-- match CausalGeometrySnapshotV1's own pydantic field names exactly (edges/
-- divergence/notes, no _json suffix) so the generic write path needs no
-- special-casing for this kind.
--
-- An earlier version of this feature had orion-field-digester write this table
-- directly via psycopg2, bypassing the bus entirely. That was corrected: the bus
-- is this repo's mechanism for tracking load/failures across services, and a new
-- write path silently bypassing it is a real observability regression, not a
-- stylistic choice.
--
-- Consumer: orion-hub (services/orion-hub/scripts/api_routes.py's
-- /api/causal-geometry/snapshot and /api/causal-geometry/history endpoints), read
-- via the hub's existing memory_pg_pool (asyncpg), the same pool already used for
-- memory cards against this same conjourney database. The column list both sides
-- agree on is a single shared constant,
-- orion.schemas.causal_geometry.CAUSAL_GEOMETRY_SNAPSHOT_SQL_COLUMNS.
--
-- Retention: not currently pruned -- `snapshot_id` is a fresh id per cycle
-- (INSERT_ONLY_MODELS fast path in orion-sql-writer), and at the producer's
-- default 24h cadence this table grows slowly. Revisit if a future rung raises
-- the cadence significantly.
--
-- On boot, orion-sql-writer applies this same DDL via `Base.metadata.create_all`
-- (services/orion-sql-writer/app/main.py). This file is the standalone equivalent
-- for manual application against a running Postgres; it is a harmless no-op once
-- applied, or once sql-writer has started at least once.

CREATE TABLE IF NOT EXISTS causal_geometry_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    designed_topology_version TEXT,
    insufficient_data BOOLEAN NOT NULL,
    edges JSONB NOT NULL,
    divergence JSONB NOT NULL,
    notes JSONB NOT NULL
);
