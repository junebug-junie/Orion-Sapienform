"""Causal Geometry v1 Phase A persistence: publish snapshots to the bus.

Earlier versions of this module wrote `CausalGeometrySnapshotV1` snapshots
directly to a dedicated Postgres table from `orion-field-digester`, bypassing
this repo's event bus entirely. That was a deliberate but wrong call -- the bus
is the "nervous system" this repo uses to track load/failures across
services, and bypassing it for a new write path is a real regression in
observability, not a stylistic choice. Corrected: this module publishes the
snapshot on `orion:causal_geometry:snapshot` (kind `causal.geometry.snapshot.v1`,
see `orion/bus/channels.yaml`); `orion-sql-writer` consumes it and writes
`causal_geometry_snapshots` via its standard `MODEL_MAP`/`DEFAULT_ROUTE_MAP`
routing (`app/models/causal_geometry_snapshot.py`), matching how every other
typed event in this repo reaches Postgres.

Uses the same one-shot connect/publish/close pattern as
`services/orion-collapse-mirror/app/routes.py`'s `log_collapse()` -- a fresh
`OrionBusAsync` per publish rather than a long-lived connection, since this is
called at most once per ~24h production cycle, not a hot path.
"""
from __future__ import annotations

import asyncio
import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.causal_geometry import CausalGeometrySnapshotV1

logger = logging.getLogger(__name__)

CAUSAL_GEOMETRY_SNAPSHOT_CHANNEL = "orion:causal_geometry:snapshot"
CAUSAL_GEOMETRY_SNAPSHOT_KIND = "causal.geometry.snapshot.v1"


async def _publish_async(
    *,
    bus_url: str,
    snapshot: CausalGeometrySnapshotV1,
    service_name: str,
    service_version: str,
    node_name: str,
) -> None:
    bus = OrionBusAsync(url=bus_url, enabled=True)
    await bus.connect()
    try:
        envelope = BaseEnvelope(
            kind=CAUSAL_GEOMETRY_SNAPSHOT_KIND,
            source=ServiceRef(name=service_name, version=service_version, node=node_name),
            payload=snapshot.model_dump(mode="json"),
        )
        await bus.publish(CAUSAL_GEOMETRY_SNAPSHOT_CHANNEL, envelope)
    finally:
        await bus.close()


def publish_snapshot(
    *,
    bus_url: str,
    bus_enabled: bool,
    snapshot: CausalGeometrySnapshotV1,
    service_name: str = "orion-field-digester",
    service_version: str = "0.1.0",
    node_name: str = "athena",
) -> dict[str, object]:
    """Publish one `CausalGeometrySnapshotV1` to the bus.

    Never raises -- mirrors the "every new adapter degrades gracefully, never
    raises" pattern used throughout this feature. Returns `{"ok": True,
    "error": None}` on success or `{"ok": False, "error": str}` on any failure
    (bus disabled, unreachable, publish rejected), so a caller can log the
    summary without a try/except of its own.

    Called from a synchronous context (this module's caller runs inside
    `asyncio.to_thread`, i.e. a worker thread with no running event loop), so
    `asyncio.run()` is used for this one-shot async operation -- safe here
    specifically because there is no already-running loop in this thread to
    conflict with.
    """
    if not bus_enabled:
        return {"ok": False, "error": "bus_disabled"}
    try:
        asyncio.run(
            _publish_async(
                bus_url=bus_url,
                snapshot=snapshot,
                service_name=service_name,
                service_version=service_version,
                node_name=node_name,
            )
        )
        return {"ok": True, "error": None}
    except Exception as exc:
        logger.warning("causal_geometry_snapshot_publish_failed: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc)}
