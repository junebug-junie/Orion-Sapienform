from __future__ import annotations

import asyncio
import json

from orion.core.bus.resilience import run_subscription, subscription_health
from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1

EQUILIBRIUM_LISTENER = "mesh-guardian-equilibrium"


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
    # down_for_ms is heartbeat lateness, 0 while heartbeats arrive. Heartbeats say
    # "degraded" while a bus listener is down, and that must not restart anything
    # (test_fresh_degraded_heartbeat_never_triggers_remediation).
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


async def watch_equilibrium(
    bus,
    channel: str,
    out_queue: asyncio.Queue[EquilibriumSnapshotV1],
    *,
    stop_event: asyncio.Event | None = None,
    name: str = EQUILIBRIUM_LISTENER,
) -> None:
    """Queue every equilibrium snapshot on ``channel``; resubscribes after a Redis drop."""

    async def handle(raw: dict) -> None:
        data = raw.get("data") if isinstance(raw, dict) else raw
        if data is None:
            return
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", "replace")
        try:
            envelope = json.loads(data) if isinstance(data, str) else data
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(envelope, dict):
            return
        if envelope.get("kind") not in ("equilibrium.snapshot.v1", "orion.equilibrium.snapshot.v1"):
            return
        payload = envelope.get("payload") or envelope
        try:
            snapshot = EquilibriumSnapshotV1.model_validate(payload)
        except Exception:
            return
        await _enqueue_snapshot(out_queue, snapshot)

    await run_subscription(bus, channel, handle, stop_event=stop_event, name=name)


def equilibrium_subscribed(name: str = EQUILIBRIUM_LISTENER) -> bool:
    """True only while the equilibrium watch holds a live subscription (not backing off)."""
    return any(row["name"] == name and row["subscribed"] for row in subscription_health()["listeners"])
