"""Mesh transport coverage: short-lived buses fold into a process-wide sink that the
publish loop drains, RpcHealthPublisher start/stop, and the default id-path normalizer."""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.http_health import http_hop_key, normalize_id_path
from orion.core.bus.rpc_health import (
    MAX_DISTINCT_HOPS,
    MAX_SAMPLES_PER_BUCKET,
    OVERFLOW_HOP_KEY,
    RpcHealthAggregator,
    SharedRpcHealthSink,
)
from orion.core.bus.rpc_health_publish import RpcHealthPublisher, rpc_health_publish_loop

SRC = ServiceRef(name="test", version="0", node="athena")


def test_absorb_is_exact_for_counts_and_hops() -> None:
    a = RpcHealthAggregator()
    b = RpcHealthAggregator()
    a.record_success(request_channel="orion:x", latency_ms=10.0)
    b.record_success(request_channel="orion:x", latency_ms=1000.0)
    b.record_timeout(request_channel="orion:y", elapsed_ms=5000.0)
    b.record_hop_success("fcc:m", 20.0)

    a.absorb(b)
    snap = a.snapshot_and_reset()

    assert snap.success_count == 2
    assert snap.timeout_count == 1
    assert snap.channel_counts == {"orion:x": 2, "orion:y": 1}
    assert snap.success_latency_ms_max == 1000.0
    assert snap.timeout_elapsed_ms_max == 5000.0
    assert snap.channel_latency["orion:x"].success_count == 2
    assert snap.channel_latency["orion:x"].max_ms == 1000.0
    assert snap.channel_latency["orion:y"].timeout_count == 1
    assert snap.channel_latency["fcc:m"].success_count == 1


def test_absorb_respects_sample_and_hop_caps() -> None:
    a = RpcHealthAggregator()
    b = RpcHealthAggregator()
    for _ in range(MAX_SAMPLES_PER_BUCKET):
        a.record_success(request_channel="orion:x", latency_ms=1.0)
    b.record_success(request_channel="orion:x", latency_ms=2.0)
    for i in range(MAX_DISTINCT_HOPS + 5):
        b.record_hop_success(f"h{i}", 1.0)

    a.absorb(b)
    snap = a.snapshot_and_reset()
    assert snap.truncated is True
    assert snap.success_count == MAX_SAMPLES_PER_BUCKET + 1  # counts stay exact
    assert len(snap.channel_latency) <= MAX_DISTINCT_HOPS + 1
    assert OVERFLOW_HOP_KEY in snap.channel_latency


def test_take_rpc_health_aggregator_resets_bus() -> None:
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    bus.record_hop_success("verb:x", 5.0)
    taken = bus.take_rpc_health_aggregator()
    assert taken.snapshot_and_reset().channel_latency["verb:x"].success_count == 1
    assert bus.get_rpc_health_snapshot().channel_latency == {}


def test_sink_folds_short_lived_bus_from_another_thread_and_loop() -> None:
    """The per-tick / per-call bus pattern: a fresh bus inside asyncio.run on a worker
    thread records an rpc outcome, is folded into the sink, and is then discarded."""
    sink = SharedRpcHealthSink()

    def worker() -> None:
        async def _tick() -> None:
            short = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
            short._rpc_health.record_success(request_channel="orion:exec:request:LLMGatewayService", latency_ms=42.0)
            short._rpc_health.record_timeout(request_channel="orion:cortex:exec:request", elapsed_ms=9000.0)
            sink.absorb_bus(short)

        asyncio.run(_tick())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    long_lived = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    sink.drain_into(long_lived._rpc_health)
    snap = long_lived.get_rpc_health_snapshot()
    assert snap.success_count == 4
    assert snap.timeout_count == 4
    assert snap.channel_latency["orion:exec:request:LLMGatewayService"].success_count == 4
    assert snap.channel_latency["orion:cortex:exec:request"].timeout_count == 4
    # Drained: a second drain adds nothing.
    sink.drain_into(long_lived._rpc_health)
    assert long_lived.get_rpc_health_snapshot().success_count == 0


def test_sink_records_hops_directly() -> None:
    sink = SharedRpcHealthSink()
    sink.record_hop_success("http:mind:8000/v1/mind/run", 12.0)
    sink.record_hop_timeout("http:mind:8000/v1/mind/run", 30000.0)
    agg = RpcHealthAggregator()
    sink.drain_into(agg)
    hop = agg.snapshot_and_reset().channel_latency["http:mind:8000/v1/mind/run"]
    assert (hop.success_count, hop.timeout_count) == (1, 1)


def test_sink_absorb_bus_without_take_method_is_noop() -> None:
    SharedRpcHealthSink().absorb_bus(object())  # never raises


@pytest.mark.asyncio
async def test_publish_loop_drains_sinks_into_published_window() -> None:
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    bus.publish = AsyncMock()  # type: ignore[method-assign]
    sink = SharedRpcHealthSink()
    sink.record_hop_success("fcc:qwen", 100.0)
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="orion:cortex:exec:request", latency_ms=50.0)
    sink.absorb_aggregator(agg)

    stop = asyncio.Event()
    task = asyncio.create_task(
        rpc_health_publish_loop(
            bus_getter=lambda: bus,
            service="svc",
            node="athena",
            instance="main",
            source=SRC,
            interval_sec=0.01,
            stop_event=stop,
            include_channel_latency=True,
            sinks=[sink],
        )
    )
    for _ in range(100):
        await asyncio.sleep(0.01)
        if bus.publish.await_count:
            break
    stop.set()
    await task

    payload = bus.publish.await_args_list[0].args[1].payload
    assert payload["success_count"] == 1
    assert payload["channel_latency"]["fcc:qwen"]["success_count"] == 1
    assert payload["channel_latency"]["orion:cortex:exec:request"]["success_count"] == 1
    assert payload["instance"] == "main"


@pytest.mark.asyncio
async def test_publisher_disabled_is_noop_and_enabled_starts_and_stops() -> None:
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    bus.publish = AsyncMock()  # type: ignore[method-assign]
    kwargs = dict(
        bus_getter=lambda: bus,
        service="svc",
        node="n",
        instance="main",
        source=SRC,
        interval_sec=0.01,
        include_channel_latency=True,
    )
    off = RpcHealthPublisher(enabled=False, **kwargs)
    assert off.start() is None
    assert off.running is False
    await off.stop()

    on = RpcHealthPublisher(enabled=True, **kwargs)
    bus.record_hop_success("verb:x", 3.0)
    assert on.start() is not None
    assert on.running
    for _ in range(100):
        await asyncio.sleep(0.01)
        if bus.publish.await_count:
            break
    await on.stop()
    assert not on.running
    assert bus.publish.await_count >= 1


def test_normalize_id_path_collapses_ids_keeps_names() -> None:
    assert normalize_id_path("/v1/runs/3f2b1c4e-1111-2222-3333-444455556666/cancel") == "/v1/runs/:id/cancel"
    assert normalize_id_path("/runs/123") == "/runs/:id"
    assert normalize_id_path("/x/deadbeefdeadbeefdead") == "/x/:id"
    assert normalize_id_path("/lanes/qwen3.5-27b/wake") == "/lanes/qwen3.5-27b/wake"
    assert normalize_id_path("/routes") == "/routes"
    assert http_hop_key("http://gw:8222/runs/42?x=1", normalize_id_path) == "http:gw:8222/runs/:id"
