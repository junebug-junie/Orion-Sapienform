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

# Single source of truth for this table's read columns (order matters -- both
# this module's own INSERT and services/orion-hub/scripts/api_routes.py's SELECT
# import this tuple directly rather than each maintaining their own literal, so
# the writer (this module, in orion-field-digester) and the reader (in
# orion-hub, a separate service/process) can never drift out of sync on column
# names/order. `orion.substrate` is already a shared package both services
# import from elsewhere in this feature (e.g. field_topology_learned_store).
SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "snapshot_id",
    "generated_at",
    "window_start",
    "window_end",
    "designed_topology_version",
    "insufficient_data",
    "edges_json",
    "divergence_json",
    "notes_json",
)

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
INSERT INTO {SNAPSHOT_TABLE} ({", ".join(SNAPSHOT_COLUMNS)})
VALUES ({", ".join(["%s"] * len(SNAPSHOT_COLUMNS))})
ON CONFLICT (snapshot_id) DO NOTHING
"""


# Set once ensure_schema() has run successfully in this process -- the DDL is
# idempotent either way, but re-issuing CREATE TABLE/INDEX IF NOT EXISTS on
# every single write (this module's only caller runs on a 24h-default cadence,
# so the cost is low, but it's still needless repeated DDL) has no benefit
# once the schema is known to exist.
_SCHEMA_ENSURED = False


def ensure_schema(cur: Any) -> None:
    """Idempotent DDL -- safe to call more than once, but `persist_snapshot()`
    only calls it once per process (see `_SCHEMA_ENSURED`)."""
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

    global _SCHEMA_ENSURED
    conn = None
    try:
        conn = psycopg2.connect(postgres_uri)
        try:
            with conn.cursor() as cur:
                if not _SCHEMA_ENSURED:
                    ensure_schema(cur)
                    _SCHEMA_ENSURED = True
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


_PRUNE_SQL = f"""
DELETE FROM {SNAPSHOT_TABLE}
WHERE snapshot_id IN (
    SELECT snapshot_id
    FROM {SNAPSHOT_TABLE}
    WHERE created_at < %s
      AND snapshot_id <> (
          SELECT snapshot_id FROM {SNAPSHOT_TABLE} ORDER BY generated_at DESC LIMIT 1
      )
    ORDER BY created_at ASC
    LIMIT %s
)
"""


def prune_snapshots(postgres_uri: str, *, retention_hours: float, batch_size: int = 500) -> dict[str, object]:
    """Delete snapshot rows older than `retention_hours`, batched, never deleting
    the single most-recent row (mirrors `services/orion-field-digester/app/store.py`'s
    `PRUNE_FIELD_STATE_SQL` guard, which keeps the hub's "Latest Snapshot" panel from
    ever going empty due to over-aggressive pruning even if retention is set very short).

    Never raises -- same degrade-gracefully posture as `persist_snapshot()`. Returns
    `{"ok": bool, "deleted": int, "error": str | None}`.
    """
    import psycopg2
    from datetime import datetime, timedelta, timezone

    if retention_hours <= 0:
        return {"ok": True, "deleted": 0, "error": None}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    conn = None
    try:
        conn = psycopg2.connect(postgres_uri)
        try:
            with conn.cursor() as cur:
                cur.execute(_PRUNE_SQL, (cutoff, batch_size))
                deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "deleted": deleted, "error": None}
    except Exception as exc:
        logger.warning("causal_geometry_snapshot_prune_failed: %s", exc, exc_info=True)
        return {"ok": False, "deleted": 0, "error": str(exc)}
