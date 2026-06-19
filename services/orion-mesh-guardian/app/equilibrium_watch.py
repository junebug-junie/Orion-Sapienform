from __future__ import annotations

import asyncio
import json

from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1


def equilibrium_status_for_service(
    snapshot: EquilibriumSnapshotV1 | None,
    *,
    heartbeat_name: str,
    grace_sec: float,
) -> tuple[bool, str | None]:
    """Return (equilibrium_bad, reason)."""
    if snapshot is None:
        return False, None
    svc = None
    for item in snapshot.services:
        if item.service == heartbeat_name:
            svc = item
            break
    if svc is None:
        if heartbeat_name in snapshot.expected_services:
            return True, "missing_from_snapshot"
        return False, None
    if svc.status == "down":
        return True, "down"
    if svc.status == "degraded" and svc.down_for_ms > grace_sec * 1000:
        return True, "degraded_beyond_grace"
    return False, None


async def _enqueue_snapshot(
    out_queue: asyncio.Queue[EquilibriumSnapshotV1],
    snapshot: EquilibriumSnapshotV1,
) -> None:
    if out_queue.full():
        try:
            out_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await out_queue.put(snapshot)


async def watch_equilibrium(bus, channel: str, out_queue: asyncio.Queue[EquilibriumSnapshotV1]) -> None:
    async with bus.subscribe(channel) as pubsub:
        async for raw in bus.iter_messages(pubsub):
            data = raw.get("data") if isinstance(raw, dict) else raw
            if data is None:
                continue
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", "replace")
            try:
                envelope = json.loads(data) if isinstance(data, str) else data
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(envelope, dict):
                continue
            if envelope.get("kind") not in ("equilibrium.snapshot.v1", "orion.equilibrium.snapshot.v1"):
                continue
            payload = envelope.get("payload") or envelope
            try:
                snapshot = EquilibriumSnapshotV1.model_validate(payload)
            except Exception:
                continue
            await _enqueue_snapshot(out_queue, snapshot)
