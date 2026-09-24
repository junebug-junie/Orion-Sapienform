"""A0 per-hop RPC health: aggregator sufficient stats, hop keys, record_hop_* API,
publisher gating, schema back-compat, and the httpx timing transport."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.http_health import (
    AsyncHopTimingTransport,
    HopTimingTransport,
    http_hop_key,
)
from orion.core.bus.rpc_health import (
    MAX_DISTINCT_HOPS,
    OVERFLOW_HOP_KEY,
    HopLatency,
    RpcHealthAggregator,
    hop_key,
)
from orion.core.bus.rpc_health_publish import (
    build_rpc_health_snapshot_envelope,
    merge_channel_latency,
    rpc_health_publish_loop,
)
from orion.schemas.telemetry.rpc_health import RpcChannelLatencyV1, RpcHealthSnapshotV1

SRC = ServiceRef(name="test", version="0", node="athena")


# --- aggregator math -------------------------------------------------------


def test_sufficient_stats_recover_exact_log_mean_and_variance() -> None:
    agg = RpcHealthAggregator()
    samples = [10.0, 100.0, 1000.0, 50.0]
    for ms in samples:
        agg.record_success(request_channel="orion:a", latency_ms=ms)
    agg.record_timeout(request_channel="orion:a", elapsed_ms=60000.0)

    hop = agg.snapshot_and_reset().channel_latency["orion:a"]
    logs = [math.log(ms) for ms in samples]
    mean = sum(logs) / len(logs)
    var = sum((x - mean) ** 2 for x in logs) / len(logs)

    assert hop.success_count == 4
    assert hop.timeout_count == 1
    assert hop.max_ms == 1000.0  # timeout elapsed never enters latency stats
    assert hop.log_ms_sum / hop.success_count == pytest.approx(mean)
    assert hop.log_ms_sumsq / hop.success_count - (hop.log_ms_sum / hop.success_count) ** 2 == pytest.approx(var)


def test_zero_and_nonfinite_latency_do_not_poison_sums() -> None:
    agg = RpcHealthAggregator()
    agg.record_hop_success("x", 0.0)
    agg.record_hop_success("x", float("nan"))
    hop = agg.snapshot_and_reset().channel_latency["x"]
    assert hop.success_count == 2
    assert math.isfinite(hop.log_ms_sum) and math.isfinite(hop.log_ms_sumsq)
    assert hop.max_ms == 0.0


def test_windows_fold_by_addition() -> None:
    """Two windows merged == one window with all samples (what a consumer relies on)."""
    a, b, both = RpcHealthAggregator(), RpcHealthAggregator(), RpcHealthAggregator()
    for ms in (5.0, 9.0):
        a.record_hop_success("h", ms)
        both.record_hop_success("h", ms)
    for ms in (300.0,):
        b.record_hop_success("h", ms)
        both.record_hop_success("h", ms)
    b.record_hop_timeout("h", 1.0)
    both.record_hop_timeout("h", 1.0)
    merged = a.snapshot_and_reset().channel_latency
    merge_channel_latency(merged, b.snapshot_and_reset().channel_latency)
    assert merged["h"].as_dict() == pytest.approx(both.snapshot_and_reset().channel_latency["h"].as_dict())


def test_health_label_keys_channel_hash_label() -> None:
    assert hop_key("orion:x", None) == "orion:x"
    assert hop_key("orion:x", "") == "orion:x"
    assert hop_key("orion:x", "log_orion_metacognition") == "orion:x#log_orion_metacognition"

    agg = RpcHealthAggregator()
    agg.record_success(request_channel="orion:x", latency_ms=5.0, health_label="log_orion_metacognition")
    agg.record_success(request_channel="orion:x", latency_ms=7.0)
    snap = agg.snapshot_and_reset()
    assert set(snap.channel_latency) == {"orion:x", "orion:x#log_orion_metacognition"}
    # pooled fields unchanged: both calls counted under the bare channel
    assert snap.success_count == 2
    assert snap.channel_counts == {"orion:x": 2}


def test_record_hop_does_not_touch_pooled_fields() -> None:
    agg = RpcHealthAggregator()
    agg.record_hop_success("verb:chat_general", 1200.0)
    agg.record_hop_timeout("governor:orion", 900000.0)
    snap = agg.snapshot_and_reset()
    assert snap.success_count == 0 and snap.timeout_count == 0
    assert snap.success_latency_ms_max is None and snap.channel_counts == {}
    assert snap.channel_latency["verb:chat_general"].success_count == 1
    assert snap.channel_latency["governor:orion"].timeout_count == 1
    assert snap.channel_latency["governor:orion"].max_ms is None


def test_hop_cardinality_overflow_is_conserved() -> None:
    agg = RpcHealthAggregator()
    for i in range(MAX_DISTINCT_HOPS + 5):
        agg.record_hop_success(f"h{i}", 1.0)
    snap = agg.snapshot_and_reset()
    assert len(snap.channel_latency) == MAX_DISTINCT_HOPS + 1
    assert snap.channel_latency[OVERFLOW_HOP_KEY].success_count == 5
    assert sum(h.success_count for h in snap.channel_latency.values()) == MAX_DISTINCT_HOPS + 5
    assert snap.truncated is True


def test_snapshot_resets_hops() -> None:
    agg = RpcHealthAggregator()
    agg.record_hop_success("h", 1.0)
    agg.snapshot_and_reset()
    assert agg.snapshot_and_reset().channel_latency == {}


# --- OrionBusAsync API -----------------------------------------------------


def test_bus_record_hop_api_lands_in_snapshot() -> None:
    bus = OrionBusAsync("redis://unused:6379/0")
    bus.record_hop_success("verb:x", 10.0)
    bus.record_hop_timeout("verb:x", None)
    hop = bus.get_rpc_health_snapshot().channel_latency["verb:x"]
    assert (hop.success_count, hop.timeout_count) == (1, 1)


class _InlinePubSub:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    async def subscribe(self, *channels: str) -> None:
        return None

    async def unsubscribe(self, *channels: str) -> None:
        return None

    async def close(self) -> None:
        return None

    async def listen(self):
        while True:
            yield await self.queue.get()


class _InlineRedis:
    def __init__(self) -> None:
        self.pubsub_obj = _InlinePubSub()

    def pubsub(self):
        return self.pubsub_obj

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_rpc_request_health_label_inline_success_and_timeout() -> None:
    bus = OrionBusAsync("redis://unused:6379/0")
    fake = _InlineRedis()

    async def _publish(channel, env):
        await fake.pubsub_obj.queue.put({"type": "message", "data": b"{}"})

    bus.publish = _publish  # type: ignore[assignment]
    env = BaseEnvelope(kind="t", source=SRC, payload={})
    with patch.object(bus, "_create_pubsub_redis", return_value=fake):
        await bus.rpc_request("orion:x", env, reply_channel="r1", timeout_sec=1.0, health_label="lbl")
        bus.publish = AsyncMock()  # no reply -> timeout
        with patch.object(bus, "_emit_rpc_timeout_grammar", AsyncMock()) as emit:
            with pytest.raises(TimeoutError):
                await bus.rpc_request("orion:x", env, reply_channel="r2", timeout_sec=0.05, health_label="lbl")
            emit.assert_awaited_once()
    hop = bus.get_rpc_health_snapshot().channel_latency["orion:x#lbl"]
    assert (hop.success_count, hop.timeout_count) == (1, 1)


# --- schema + publisher ----------------------------------------------------


def _old_payload() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "service": "cortex-exec",
        "node": "athena",
        "instance": None,
        "window_start": now,
        "window_end": now,
        "success_count": 1,
        "timeout_count": 0,
        "success_latency_ms_p50": 1.0,
        "success_latency_ms_p95": 1.0,
        "success_latency_ms_max": 1.0,
        "timeout_elapsed_ms_max": None,
        "channel_counts": {"orion:a": 1},
        "truncated": False,
    }


def test_schema_accepts_old_payload_without_channel_latency() -> None:
    m = RpcHealthSnapshotV1.model_validate(_old_payload())
    assert m.channel_latency == {}


def test_schema_round_trip_with_channel_latency() -> None:
    payload = _old_payload()
    payload["channel_latency"] = {
        "verb:x": {"success_count": 2, "timeout_count": 1, "log_ms_sum": 4.0, "log_ms_sumsq": 8.5, "max_ms": 9.0}
    }
    m = RpcHealthSnapshotV1.model_validate(payload)
    assert RpcHealthSnapshotV1.model_validate(m.model_dump(mode="json")) == m
    with pytest.raises(Exception):
        RpcChannelLatencyV1.model_validate({"success_count": 1, "bogus": 1})


def test_registry_resolves_new_model() -> None:
    from orion.schemas.registry import resolve

    assert resolve("RpcChannelLatencyV1") is RpcChannelLatencyV1


def _snap_with_hop():
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="orion:a", latency_ms=3.0)
    agg.record_hop_success("verb:x", 20.0)
    return agg.snapshot_and_reset()


def test_envelope_omits_channel_latency_key_when_disabled() -> None:
    env = build_rpc_health_snapshot_envelope(
        _snap_with_hop(), service="s", node="n", instance="i", source=SRC
    )
    assert "channel_latency" not in env.payload
    assert env.payload["instance"] == "i"


def test_envelope_includes_channel_latency_when_enabled() -> None:
    env = build_rpc_health_snapshot_envelope(
        _snap_with_hop(), service="s", node="n", instance=None, source=SRC, include_channel_latency=True
    )
    cl = env.payload["channel_latency"]
    assert set(cl) == {"orion:a", "verb:x"}
    assert cl["verb:x"]["success_count"] == 1
    RpcHealthSnapshotV1.model_validate(env.payload)


@pytest.mark.asyncio
async def test_publish_loop_merges_hop_only_bus_channel_latency_not_pooled() -> None:
    main = OrionBusAsync("redis://unused:6379/0")
    side = OrionBusAsync("redis://unused:6379/0")
    main._rpc_health.record_success(request_channel="orion:a", latency_ms=3.0)
    side._rpc_health.record_success(request_channel="orion:b", latency_ms=50.0, health_label="m")
    main.publish = AsyncMock()
    stop = asyncio.Event()

    task = asyncio.create_task(
        rpc_health_publish_loop(
            bus_getter=lambda: main,
            service="s",
            node="n",
            instance="chat",
            source=SRC,
            interval_sec=0.02,
            stop_event=stop,
            include_channel_latency=True,
            hop_only_bus_getters=[lambda: side, lambda: None, lambda: main],
        )
    )
    for _ in range(100):
        if main.publish.await_count:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await task
    payload = main.publish.await_args_list[0].args[1].payload
    assert payload["success_count"] == 1  # side bus pooled NOT merged
    assert payload["channel_counts"] == {"orion:a": 1}
    assert set(payload["channel_latency"]) == {"orion:a", "orion:b#m"}
    assert payload["instance"] == "chat"


# --- http timing transport -------------------------------------------------


def test_http_hop_key() -> None:
    assert http_hop_key("http://orion-mind:8080/v1/run?x=1") == "http:orion-mind:8080/v1/run"
    assert http_hop_key("https://gw/v1/chat") == "http:gw/v1/chat"
    assert http_hop_key("http://h/runs/abc", lambda p: "/runs/:id") == "http:h/runs/:id"


@pytest.mark.asyncio
async def test_async_transport_records_success_504_and_timeout() -> None:
    agg = RpcHealthAggregator()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/slow":
            raise httpx.ReadTimeout("t", request=request)
        if request.url.path == "/gw":
            return httpx.Response(504)
        return httpx.Response(500 if request.url.path == "/err" else 200)

    transport = AsyncHopTimingTransport(httpx.MockTransport(handler), recorder_getter=lambda: agg)
    async with httpx.AsyncClient(transport=transport) as client:
        await client.get("http://h/ok")
        await client.get("http://h/err")
        await client.get("http://h/gw")
        with pytest.raises(httpx.ReadTimeout):
            await client.get("http://h/slow")
    cl = agg.snapshot_and_reset().channel_latency
    assert cl["http:h/ok"].success_count == 1
    assert cl["http:h/err"].success_count == 1  # app error is not a transport failure
    assert cl["http:h/gw"].timeout_count == 1
    assert cl["http:h/slow"].timeout_count == 1


def test_sync_transport_records_and_recorder_failure_never_raises() -> None:
    agg = RpcHealthAggregator()
    t = HopTimingTransport(httpx.MockTransport(lambda r: httpx.Response(200)), recorder_getter=lambda: agg)
    with httpx.Client(transport=t) as client:
        client.get("http://h/ok")
    assert agg.snapshot_and_reset().channel_latency["http:h/ok"].success_count == 1

    broken = MagicMock()
    broken.record_hop_success.side_effect = RuntimeError("boom")
    t2 = HopTimingTransport(httpx.MockTransport(lambda r: httpx.Response(200)), recorder_getter=lambda: broken)
    with httpx.Client(transport=t2) as client:
        assert client.get("http://h/ok").status_code == 200
    t3 = HopTimingTransport(httpx.MockTransport(lambda r: httpx.Response(200)), recorder_getter=lambda: None)
    with httpx.Client(transport=t3) as client:
        assert client.get("http://h/ok").status_code == 200


def test_hop_latency_as_dict_matches_schema_fields() -> None:
    assert set(HopLatency().as_dict()) == set(RpcChannelLatencyV1.model_fields)
