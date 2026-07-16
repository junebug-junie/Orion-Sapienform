"""Causal Geometry v1, Phase A persistence: durable storage for
`CausalGeometrySnapshotV1` snapshots.

The design spec (`docs/superpowers/specs/2026-07-16-causal-geometry-v1-design.md`)
called for Phase A snapshots to be persisted "via sql-writer", which would route
through `services/orion-sql-writer/app/worker.py`'s shared 2300-line worker used
by a dozen unrelated pipelines. After investigation, that path was judged too
invasive for the value here. Instead this module mirrors
`services/orion-field-digester/app/store.py`'s `FieldDigesterStore` pattern
(and `orion/substrate/causal_geometry_engine.py`'s `fetch_channels()` connection
handling): a small, dedicated table this feature owns and writes to itself
directly via psycopg2, using the same `postgres_uri` setting the engine already
reads channels from.

This is what lets the hub's Snapshot/History API endpoints stop being
permanently `degraded: True` -- there is finally a durable row to read back.

Like every other adapter in this feature (see
`orion/substrate/field_topology_learned_store.py`'s sqlite fallback and
`services/orion-field-digester/app/digestion/diffusion.py`'s
`_load_learned_overlay()`), persistence here never raises. A failed write
degrades to a logged warning and a returned `{"ok": False, "error": ...}` --
never an exception that could take down the measurement/proposal cycle that
called it.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from orion.schemas.causal_geometry import CausalGeometrySnapshotV1

logger = logging.getLogger(__name__)

SNAPSHOT_TABLE = "causal_geometry_snapshots"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SNAPSHOT_TABLE} (
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
)
"""

_CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_causal_geometry_snapshots_generated_at
    ON {SNAPSHOT_TABLE} (generated_at DESC)
"""

_INSERT_SQL = f"""
INSERT INTO {SNAPSHOT_TABLE} (
    snapshot_id,
    generated_at,
    window_start,
    window_end,
    designed_topology_version,
    insufficient_data,
    edges_json,
    divergence_json,
    notes_json
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (snapshot_id) DO NOTHING
"""


def ensure_schema(cur: Any) -> None:
    """Idempotent DDL -- safe to call on every write."""
    cur.execute(_CREATE_TABLE_SQL)
    cur.execute(_CREATE_INDEX_SQL)


def persist_snapshot(postgres_uri: str, snapshot: CausalGeometrySnapshotV1) -> dict[str, object]:
    """Persist one `CausalGeometrySnapshotV1` row.

    Never raises -- mirrors the "every new adapter degrades gracefully, never
    raises" pattern used throughout this feature. Returns
    `{"ok": True}` on success or `{"ok": False, "error": str}` on any failure
    (connection, schema, or insert), so a caller can log the summary without a
    try/except of its own.
    """
    import psycopg2

    conn = None
    try:
        conn = psycopg2.connect(postgres_uri)
        try:
            with conn.cursor() as cur:
                ensure_schema(cur)
                cur.execute(
                    _INSERT_SQL,
                    (
                        snapshot.snapshot_id,
                        snapshot.generated_at,
                        snapshot.window_start,
                        snapshot.window_end,
                        snapshot.designed_topology_version,
                        snapshot.insufficient_data,
                        json.dumps([e.model_dump(mode="json") for e in snapshot.edges], default=str),
                        json.dumps([d.model_dump(mode="json") for d in snapshot.divergence], default=str),
                        json.dumps(snapshot.notes),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "error": None}
    except Exception as exc:
        logger.warning("causal_geometry_snapshot_persist_failed: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc)}
