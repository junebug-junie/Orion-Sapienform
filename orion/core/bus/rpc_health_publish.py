"""Shared periodic RPC-health snapshot publisher.

Step 3 of docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md. Not
part of BaseChassis._heartbeat_loop() -- deliberately independent, because a service's real
RPC-calling OrionBusAsync instance can be a fork_rpc_client() child rather than the chassis's
own `svc.bus` (confirmed for orion-cortex-exec and orion-cortex-orch in that spec's "Resolved
(2026-07-24)" section). Draining the wrong instance would silently report an always-empty
aggregator, so this helper takes a `bus_getter` callback rather than a bus instance, and calls
it fresh on every tick rather than capturing one bus reference at startup.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.rpc_health import RpcHealthSnapshot
from orion.schemas.telemetry.rpc_health import RpcHealthSnapshotV1

logger = logging.getLogger("orion.bus.rpc_health_publish")

RPC_HEALTH_SNAPSHOT_CHANNEL = "orion:rpc_health:snapshot"
RPC_HEALTH_SNAPSHOT_KIND = "rpc_health.snapshot.v1"


def build_rpc_health_snapshot_envelope(
    snapshot: RpcHealthSnapshot,
    *,
    service: str,
    node: Optional[str],
    instance: Optional[str],
    source: ServiceRef,
) -> BaseEnvelope:
    payload = RpcHealthSnapshotV1(
        service=service,
        node=node,
        instance=instance,
        window_start=snapshot.window_start,
        window_end=snapshot.window_end,
        success_count=snapshot.success_count,
        timeout_count=snapshot.timeout_count,
        success_latency_ms_p50=snapshot.success_latency_ms_p50,
        success_latency_ms_p95=snapshot.success_latency_ms_p95,
        success_latency_ms_max=snapshot.success_latency_ms_max,
        timeout_elapsed_ms_max=snapshot.timeout_elapsed_ms_max,
        channel_counts=dict(snapshot.channel_counts),
        truncated=snapshot.truncated,
    )
    return BaseEnvelope(
        kind=RPC_HEALTH_SNAPSHOT_KIND,
        source=source,
        payload=payload.model_dump(mode="json"),
    )


async def rpc_health_publish_loop(
    *,
    bus_getter: Callable[[], OrionBusAsync],
    service: str,
    node: Optional[str],
    instance: Optional[str],
    source: ServiceRef,
    interval_sec: float,
    stop_event: asyncio.Event,
) -> None:
    """Sleeps interval_sec, then drains bus_getter()'s current RPC-health snapshot and
    publishes it. Never raises past this loop -- a publish failure is logged and the loop
    continues, since this is telemetry, not a path any real turn depends on."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            return  # stop_event was set during the sleep
        except asyncio.TimeoutError:
            pass
        try:
            bus = bus_getter()
            snapshot = bus.get_rpc_health_snapshot()
            envelope = build_rpc_health_snapshot_envelope(
                snapshot,
                service=service,
                node=node,
                instance=instance,
                source=source,
            )
            await bus.publish(RPC_HEALTH_SNAPSHOT_CHANNEL, envelope)
        except Exception:
            logger.warning("rpc_health_publish_failed service=%s", service, exc_info=True)
