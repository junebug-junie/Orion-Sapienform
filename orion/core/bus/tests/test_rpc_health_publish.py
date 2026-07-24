"""Unit tests for orion/core/bus/rpc_health_publish.py.

Step 3 of docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.rpc_health import RpcHealthAggregator
from orion.core.bus.rpc_health_publish import (
    RPC_HEALTH_SNAPSHOT_CHANNEL,
    RPC_HEALTH_SNAPSHOT_KIND,
    build_rpc_health_snapshot_envelope,
    rpc_health_publish_loop,
)


def _source() -> ServiceRef:
    return ServiceRef(name="cortex-exec", node="athena")


def test_build_envelope_carries_all_snapshot_fields() -> None:
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="orion:cortex:exec:request", latency_ms=42.0)
    agg.record_timeout(request_channel="orion:cortex:exec:request", elapsed_ms=6000.0)
    snapshot = agg.snapshot_and_reset()

    env = build_rpc_health_snapshot_envelope(
        snapshot, service="cortex-exec", node="athena", instance=None, source=_source()
    )

    assert env.kind == RPC_HEALTH_SNAPSHOT_KIND
    assert env.payload["service"] == "cortex-exec"
    assert env.payload["success_count"] == 1
    assert env.payload["timeout_count"] == 1
    assert env.payload["channel_counts"] == {"orion:cortex:exec:request": 2}


def test_build_envelope_on_empty_snapshot_is_not_degenerate_shaped() -> None:
    """An empty snapshot still carries a real, well-formed window -- zero counts are a
    real answer, not a missing-field placeholder."""
    agg = RpcHealthAggregator()
    snapshot = agg.snapshot_and_reset()
    env = build_rpc_health_snapshot_envelope(
        snapshot, service="cortex-orch", node="athena", instance=None, source=_source()
    )
    assert env.payload["success_count"] == 0
    assert env.payload["timeout_count"] == 0
    assert "window_start" in env.payload
    assert "window_end" in env.payload


@pytest.mark.asyncio
async def test_publish_loop_resolves_bus_getter_fresh_each_tick_not_captured_once() -> None:
    """Mirrors the real shape of _bus_for_rpc() in cortex-exec/cortex-orch: returns the
    wrong (idle) bus until an external mutation flips which instance is live, then
    returns the real one. A prior version of this test used bus_getter=lambda: real_bus,
    which returns the SAME object every call and would have passed identically even if
    the implementation captured `bus = bus_getter()` once before the loop instead of
    calling it fresh every tick -- it never actually distinguished the two
    implementations. This version does: if the loop captured its bus reference once at
    start (when current_bus['bus'] is still idle_bus), idle_bus would receive every
    publish and real_bus would receive none."""
    idle_bus = MagicMock()
    idle_bus.get_rpc_health_snapshot = MagicMock(
        return_value=RpcHealthAggregator().snapshot_and_reset()
    )
    idle_bus.publish = AsyncMock()

    real_bus = MagicMock()
    real_bus.get_rpc_health_snapshot = MagicMock(
        return_value=RpcHealthAggregator().snapshot_and_reset()
    )
    real_bus.publish = AsyncMock()

    current_bus = {"bus": idle_bus}

    def _bus_getter():
        return current_bus["bus"]

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        rpc_health_publish_loop(
            bus_getter=_bus_getter,
            service="cortex-exec",
            node="athena",
            instance=None,
            source=_source(),
            interval_sec=0.01,
            stop_event=stop_event,
        )
    )
    await asyncio.sleep(0.015)  # let at least one tick land against idle_bus
    current_bus["bus"] = real_bus  # simulate _rpc_bus becoming the live instance mid-run
    await asyncio.sleep(0.03)  # let at least one tick land against real_bus
    stop_event.set()
    await task

    assert idle_bus.publish.await_count >= 1
    assert real_bus.publish.await_count >= 1
    published_channel = real_bus.publish.await_args_list[0].args[0]
    assert published_channel == RPC_HEALTH_SNAPSHOT_CHANNEL


@pytest.mark.asyncio
async def test_publish_loop_survives_publish_failure_and_keeps_ticking() -> None:
    bus = MagicMock()
    bus.get_rpc_health_snapshot = MagicMock(
        return_value=RpcHealthAggregator().snapshot_and_reset()
    )
    bus.publish = AsyncMock(side_effect=[RuntimeError("boom"), None, None])

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        rpc_health_publish_loop(
            bus_getter=lambda: bus,
            service="cortex-exec",
            node="athena",
            instance=None,
            source=_source(),
            interval_sec=0.01,
            stop_event=stop_event,
        )
    )
    await asyncio.sleep(0.05)
    stop_event.set()
    await task

    assert bus.publish.await_count >= 2


@pytest.mark.asyncio
async def test_publish_loop_stops_promptly_when_stop_event_set_during_sleep() -> None:
    bus = MagicMock()
    bus.get_rpc_health_snapshot = MagicMock(
        return_value=RpcHealthAggregator().snapshot_and_reset()
    )
    bus.publish = AsyncMock()

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        rpc_health_publish_loop(
            bus_getter=lambda: bus,
            service="cortex-exec",
            node="athena",
            instance=None,
            source=_source(),
            interval_sec=30.0,
            stop_event=stop_event,
        )
    )
    await asyncio.sleep(0.01)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert bus.publish.await_count == 0
