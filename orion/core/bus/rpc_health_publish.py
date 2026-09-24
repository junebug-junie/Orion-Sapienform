"""Shared periodic RPC-health snapshot publisher.

Step 3 of docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md. Not
part of BaseChassis._heartbeat_loop() -- deliberately independent, because a service's real
RPC-calling OrionBusAsync instance can be a fork_rpc_client() child rather than the chassis's
own `svc.bus` (confirmed for orion-cortex-exec and orion-cortex-orch in that spec's "Resolved
(2026-07-24)" section). Draining the wrong instance would silently report an always-empty
aggregator, so this helper takes a `bus_getter` callback rather than a bus instance, and calls
it fresh on every tick rather than capturing one bus reference at startup.

Per-hop breakdown (``channel_latency``, 2026-09-24): emitted only when the caller passes
``include_channel_latency=True`` (each service wires this to its own
``RPC_HEALTH_CHANNEL_LATENCY_ENABLED``). ``RpcHealthSnapshotV1`` is ``extra="forbid"``, so
this stays off until every consumer validating the payload is running a build that knows
the field -- consumer-first rollout.

``hop_only_bus_getters``: extra OrionBusAsync instances in the same process whose
aggregators are drained each tick and whose ``channel_latency`` is folded into the
published snapshot -- their POOLED fields are discarded, so the pooled fields keep their
meaning of "rpc_request() outcomes on the primary bus". For services where some RPC
runs on a second bus (e.g. cortex-orch's metacog dispatch on the equilibrium Hunter's
bus) and should be visible per-hop without changing the pooled numbers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, Iterable, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.rpc_health import HopLatency, RpcHealthSnapshot, SharedRpcHealthSink
from orion.schemas.telemetry.rpc_health import RpcChannelLatencyV1, RpcHealthSnapshotV1

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
    include_channel_latency: bool = False,
) -> BaseEnvelope:
    channel_latency = (
        {k: RpcChannelLatencyV1(**v.as_dict()) for k, v in snapshot.channel_latency.items()}
        if include_channel_latency
        else {}
    )
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
        channel_latency=channel_latency,
    )
    dumped = payload.model_dump(mode="json")
    if not include_channel_latency:
        # Byte-identical to the pre-A0 payload: a consumer still on an old
        # extra="forbid" build must never see the key at all, not even as {}.
        dumped.pop("channel_latency", None)
    return BaseEnvelope(
        kind=RPC_HEALTH_SNAPSHOT_KIND,
        source=source,
        payload=dumped,
    )


def merge_channel_latency(into: Dict[str, HopLatency], other: Dict[str, HopLatency]) -> None:
    """Fold ``other``'s per-hop sufficient statistics into ``into`` in place. Exact:
    counts and log sums add, max takes the larger."""
    for key, stats in other.items():
        cur = into.get(key)
        if cur is None:
            into[key] = HopLatency(**stats.as_dict())
            continue
        cur.success_count += stats.success_count
        cur.timeout_count += stats.timeout_count
        cur.log_ms_sum += stats.log_ms_sum
        cur.log_ms_sumsq += stats.log_ms_sumsq
        if stats.max_ms is not None and (cur.max_ms is None or stats.max_ms > cur.max_ms):
            cur.max_ms = stats.max_ms


async def rpc_health_publish_loop(
    *,
    bus_getter: Callable[[], OrionBusAsync],
    service: str,
    node: Optional[str],
    instance: Optional[str],
    source: ServiceRef,
    interval_sec: float,
    stop_event: asyncio.Event,
    include_channel_latency: bool = False,
    hop_only_bus_getters: Iterable[Callable[[], Optional[OrionBusAsync]]] = (),
    sinks: Iterable[SharedRpcHealthSink] = (),
) -> None:
    """Sleeps interval_sec, then drains bus_getter()'s current RPC-health snapshot and
    publishes it. Never raises past this loop -- a publish failure is logged and the loop
    continues, since this is telemetry, not a path any real turn depends on."""
    hop_only_getters = list(hop_only_bus_getters)
    sink_list = list(sinks)
    # Discard whatever hop-only buses accumulated before the loop started, so the first
    # published window covers one interval, not "since process start".
    for extra_getter in hop_only_getters:
        try:
            extra = extra_getter()
            if extra is not None:
                extra.get_rpc_health_snapshot()
        except Exception:
            logger.warning("rpc_health_hop_only_initial_drain_failed service=%s", service, exc_info=True)
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            return  # stop_event was set during the sleep
        except asyncio.TimeoutError:
            pass
        try:
            bus = bus_getter()
            for sink in sink_list:
                sink.drain_into(bus._rpc_health)
            snapshot = bus.get_rpc_health_snapshot()
            for extra_getter in hop_only_getters:
                try:
                    extra = extra_getter()
                    if extra is None or extra is bus:
                        continue
                    merge_channel_latency(
                        snapshot.channel_latency, extra.get_rpc_health_snapshot().channel_latency
                    )
                except Exception:
                    logger.warning("rpc_health_hop_only_drain_failed service=%s", service, exc_info=True)
            envelope = build_rpc_health_snapshot_envelope(
                snapshot,
                service=service,
                node=node,
                instance=instance,
                source=source,
                include_channel_latency=include_channel_latency,
            )
            await bus.publish(RPC_HEALTH_SNAPSHOT_CHANNEL, envelope)
        except Exception:
            logger.warning("rpc_health_publish_failed service=%s", service, exc_info=True)


class RpcHealthPublisher:
    """Start/stop wrapper around ``rpc_health_publish_loop`` so each service's wiring is
    a few lines: construct it, ``start()`` once its long-lived bus is connected,
    ``await stop()`` on shutdown. ``start()`` is a no-op when ``enabled`` is false.

    ``bus_getter`` must return a CONNECTED, long-lived bus (publish goes through it).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        bus_getter: Callable[[], Optional[OrionBusAsync]],
        service: str,
        node: Optional[str],
        instance: Optional[str],
        source: ServiceRef,
        interval_sec: float,
        include_channel_latency: bool,
        sinks: Iterable[SharedRpcHealthSink] = (),
        hop_only_bus_getters: Iterable[Callable[[], Optional[OrionBusAsync]]] = (),
    ) -> None:
        self.enabled = bool(enabled)
        self._bus_getter = bus_getter
        self._kwargs = dict(
            service=service,
            node=node,
            instance=instance,
            source=source,
            interval_sec=float(interval_sec),
            include_channel_latency=bool(include_channel_latency),
            sinks=tuple(sinks),
            hop_only_bus_getters=tuple(hop_only_bus_getters),
        )
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> Optional[asyncio.Task]:
        if not self.enabled or self.running:
            return self._task
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(
            rpc_health_publish_loop(bus_getter=self._bus_getter, stop_event=self._stop, **self._kwargs),
            name="rpc-health-publish",
        )
        logger.info(
            "rpc_health_publish_started service=%s instance=%s interval=%ss channel=%s channel_latency=%s",
            self._kwargs["service"],
            self._kwargs["instance"],
            self._kwargs["interval_sec"],
            RPC_HEALTH_SNAPSHOT_CHANNEL,
            self._kwargs["include_channel_latency"],
        )
        return self._task

    async def stop(self, timeout_sec: float = 5.0) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(task, timeout=timeout_sec)
        except Exception:
            task.cancel()
