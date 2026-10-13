"""Degraded RPC replies (2026-10-13): a reply the replier marked as not its real answer
is counted in degraded_count and never timed as the hop's speed.

Live cause: on 2026-10-08 every journal.compose call on cortex-orch's background lane
was a gpu_pool_unavailable refusal answered in ~0.3-1 s, recorded as a success, and the
transport baseline's floor learned ~1.3 s for a ~8-15 s hop.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import Rabbit, _stamp_degraded_reply
from orion.core.bus.codec import OrionCodec
from orion.core.bus.rpc_degraded import (
    DEGRADED_TRACE_KEY,
    MAX_REASON_CHARS,
    begin_collecting,
    current_degraded_reason,
    degraded_reason,
    end_collecting,
    mark_degraded,
    note_degraded,
    strip_degraded,
)
from orion.core.bus.rpc_health import RpcHealthAggregator, SharedRpcHealthSink
from orion.core.bus.rpc_health_publish import build_rpc_health_snapshot_envelope, merge_channel_latency
from orion.schemas.telemetry.rpc_health import RpcChannelLatencyV1, RpcHealthSnapshotV1

SRC = ServiceRef(name="test", version="0", node="athena")


def _env(kind: str = "cortex.exec.result", trace: dict | None = None) -> BaseEnvelope:
    return BaseEnvelope(kind=kind, source=SRC, trace=trace or {}, payload={"ok": True})


# --- the marker ------------------------------------------------------------


def test_explicit_trace_marker_and_system_error_kind_are_degraded() -> None:
    assert degraded_reason(_env(trace={DEGRADED_TRACE_KEY: "gpu_pool_unavailable"})) == "gpu_pool_unavailable"
    assert degraded_reason(_env(kind="system.error")) == "system_error"
    assert degraded_reason(_env(kind="system.error.v1")) == "system_error"


def test_a_real_reply_is_not_degraded_and_nothing_is_inferred_from_content() -> None:
    assert degraded_reason(_env()) is None
    # A payload that *looks* like a failure is the replier's business to mark; the
    # caller never guesses from content.
    looks_failed = BaseEnvelope(kind="llm.chat.result", source=SRC, payload={"content": "", "raw": {"error": "x"}})
    assert degraded_reason(looks_failed) is None
    assert degraded_reason(_env(trace={DEGRADED_TRACE_KEY: "   "})) is None
    assert degraded_reason(_env(trace={DEGRADED_TRACE_KEY: 7})) is None
    assert degraded_reason(None) is None


def test_a_marker_echoed_from_the_request_trace_is_not_the_replys_verdict() -> None:
    req_trace = {DEGRADED_TRACE_KEY: "upstream"}
    echoed = _env(trace=dict(req_trace))
    assert degraded_reason(echoed, request_trace=req_trace) is None
    own = _env(trace={DEGRADED_TRACE_KEY: "gpu_pool_unavailable"})
    assert degraded_reason(own, request_trace=req_trace) == "gpu_pool_unavailable"


def test_mark_degraded_copies_keeps_first_cause_and_bounds_the_reason() -> None:
    env = _env(trace={"trace_id": "t"})
    marked = mark_degraded(env, "first")
    assert env.trace == {"trace_id": "t"}  # frozen envelope: a copy, not a mutation
    assert marked.trace == {"trace_id": "t", DEGRADED_TRACE_KEY: "first"}
    assert mark_degraded(marked, "second").trace[DEGRADED_TRACE_KEY] == "first"
    assert mark_degraded(env, "  ") is env
    assert len(mark_degraded(env, "x" * 500).trace[DEGRADED_TRACE_KEY]) == MAX_REASON_CHARS
    assert strip_degraded(marked.trace) == {"trace_id": "t"}
    assert strip_degraded(None) == {}


def test_note_is_a_noop_outside_a_collector_and_collects_inside() -> None:
    note_degraded("nobody listening")
    assert current_degraded_reason() is None
    token = begin_collecting()
    try:
        note_degraded("a")
        note_degraded("b")
        note_degraded("a")
        assert current_degraded_reason() == "a"
    finally:
        end_collecting(token)
    assert current_degraded_reason() is None


@pytest.mark.asyncio
async def test_a_note_from_a_child_task_reaches_the_request_collector() -> None:
    token = begin_collecting()
    try:
        await asyncio.create_task(asyncio.sleep(0, result=note_degraded("from_task")))
        async def _child() -> None:
            note_degraded("from_child")
        await asyncio.gather(_child())
        assert current_degraded_reason() in {"from_task", "from_child"}
    finally:
        end_collecting(token)


# --- the Rabbit chassis stamps the reply -----------------------------------


class _CaptureBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, msg: BaseEnvelope) -> None:
        self.published.append((channel, msg))


def _rabbit(handler) -> tuple[Rabbit, _CaptureBus]:
    r = Rabbit.__new__(Rabbit)
    r.handler = handler
    r.bus = _CaptureBus()

    class _Cfg:
        service_name = "test-svc"

    r.cfg = _Cfg()
    return r, r.bus


@pytest.mark.asyncio
async def test_rabbit_stamps_a_reply_whose_handling_noted_degradation() -> None:
    async def handler(env: BaseEnvelope) -> BaseEnvelope:
        note_degraded("llm_gateway:gpu_pool_unavailable:deadline")
        return _env()

    rabbit, bus = _rabbit(handler)
    req = BaseEnvelope(kind="cortex.exec.request", source=SRC, reply_to="orion:r", payload={})
    await rabbit._handle_decoded_envelope(channel="orion:x", env=req)
    (_, reply), = bus.published
    assert degraded_reason(reply) == "llm_gateway:gpu_pool_unavailable:deadline"
    assert current_degraded_reason() is None  # collector closed after the request


@pytest.mark.asyncio
async def test_rabbit_leaves_a_clean_reply_alone_and_strips_an_inherited_marker() -> None:
    async def clean(env: BaseEnvelope) -> BaseEnvelope:
        return _env(trace={"trace_id": "t"})

    rabbit, bus = _rabbit(clean)
    req = BaseEnvelope(kind="k", source=SRC, reply_to="orion:r", payload={})
    await rabbit._handle_decoded_envelope(channel="orion:x", env=req)
    assert degraded_reason(bus.published[-1][1]) is None

    async def echo(env: BaseEnvelope) -> BaseEnvelope:
        return _env(trace=dict(env.trace))  # copies the request trace, as _derive_envelope did

    rabbit, bus = _rabbit(echo)
    req = BaseEnvelope(kind="k", source=SRC, reply_to="orion:r", trace={DEGRADED_TRACE_KEY: "old"}, payload={})
    await rabbit._handle_decoded_envelope(channel="orion:x", env=req)
    assert DEGRADED_TRACE_KEY not in bus.published[-1][1].trace


def test_stamp_never_raises_on_a_non_envelope() -> None:
    assert _stamp_degraded_reply("not an envelope", request=_env(), noted="x") == "not an envelope"


# --- aggregator: counted, never timed --------------------------------------


def test_degraded_is_counted_but_never_enters_latency_statistics() -> None:
    agg = RpcHealthAggregator()
    for ms in (8000.0, 12000.0):
        agg.record_success(request_channel="orion:bg", latency_ms=ms)
    for _ in range(30):
        agg.record_degraded(request_channel="orion:bg", latency_ms=300.0, reason="gpu_pool_unavailable")
    snap = agg.snapshot_and_reset()
    hop = snap.channel_latency["orion:bg"]
    assert (hop.success_count, hop.timeout_count, hop.degraded_count) == (2, 0, 30)
    assert hop.log_ms_sum / hop.success_count == pytest.approx((math.log(8000) + math.log(12000)) / 2)
    assert hop.max_ms == 12000.0
    # pooled: same separation
    assert (snap.success_count, snap.degraded_count) == (2, 30)
    assert snap.success_latency_ms_p50 == pytest.approx(10000.0)
    assert snap.channel_counts["orion:bg"] == 32  # every call, all outcomes
    after = agg.snapshot_and_reset()
    assert after.degraded_count == 0 and after.channel_latency == {}


def test_degraded_respects_health_label_and_hand_rolled_hops() -> None:
    agg = RpcHealthAggregator()
    agg.record_degraded(request_channel="orion:bg", health_label="lbl")
    agg.record_hop_degraded("verb:journal.compose", 250.0)
    snap = agg.snapshot_and_reset()
    assert snap.channel_latency["orion:bg#lbl"].degraded_count == 1
    assert snap.channel_latency["verb:journal.compose"].degraded_count == 1
    assert snap.channel_latency["verb:journal.compose"].max_ms is None
    assert snap.degraded_count == 1  # hand-rolled hops never touch pooled fields


def test_absorb_merge_and_sink_add_degraded_exactly() -> None:
    a, b = RpcHealthAggregator(), RpcHealthAggregator()
    a.record_success(request_channel="h", latency_ms=10.0)
    a.record_degraded(request_channel="h")
    b.record_degraded(request_channel="h")
    b.record_success(request_channel="h", latency_ms=20.0)
    a.absorb(b)
    snap = a.snapshot_and_reset()
    hop = snap.channel_latency["h"]
    assert (hop.success_count, hop.degraded_count, snap.degraded_count) == (2, 2, 2)
    assert hop.log_ms_sum == pytest.approx(math.log(10.0) + math.log(20.0))

    into = {"h": hop}
    c = RpcHealthAggregator()
    c.record_hop_degraded("h")
    c.record_hop_degraded("new")
    merge_channel_latency(into, c.snapshot_and_reset().channel_latency)
    assert into["h"].degraded_count == 3 and into["new"].degraded_count == 1

    sink = SharedRpcHealthSink()
    sink.record_hop_degraded("verb:x", 5.0)
    target = RpcHealthAggregator()
    sink.drain_into(target)
    assert target.snapshot_and_reset().channel_latency["verb:x"].degraded_count == 1


# --- rpc_request classifies the reply ---------------------------------------


@pytest.mark.asyncio
async def test_rpc_request_worker_path_records_a_marked_reply_as_degraded() -> None:
    bus = OrionBusAsync("redis://unused:6379/0")
    bus._rpc_worker_task = asyncio.get_running_loop().create_future()  # "worker running"
    bus._rpc_subscribe = AsyncMock()
    bus.rpc_release_reply_channel = AsyncMock()
    codec = OrionCodec()
    replies = iter(
        [
            mark_degraded(_env(), "gpu_pool_unavailable"),
            _env(),
            _env(kind="system.error"),
        ]
    )

    async def _publish(channel: str, env: BaseEnvelope) -> None:
        reply = next(replies).model_copy(update={"correlation_id": env.correlation_id})
        await bus._handle_rpc_result({"channel": env.reply_to.encode(), "data": codec.encode(reply)})

    bus.publish = _publish  # type: ignore[assignment]
    for i in range(3):
        req = BaseEnvelope(kind="cortex.exec.request", source=SRC, reply_to=f"orion:r{i}", payload={})
        msg = await bus.rpc_request("orion:bg", req, reply_channel=f"orion:r{i}", timeout_sec=1.0)
        assert msg["data"]  # the caller still gets the raw reply, unchanged
    assert bus._rpc_reply_envelopes == {}  # nothing left behind
    hop = bus.get_rpc_health_snapshot().channel_latency["orion:bg"]
    assert (hop.success_count, hop.degraded_count) == (1, 2)
    bus._rpc_worker_task.cancel()


@pytest.mark.asyncio
async def test_rpc_request_worker_path_never_leaks_a_stashed_reply() -> None:
    """A reply can land (stashed for classification) and the call still fail before it
    is read -- here the publish raises after the reply arrived. The stash must not grow."""
    bus = OrionBusAsync("redis://unused:6379/0")
    bus._rpc_worker_task = asyncio.get_running_loop().create_future()
    bus._rpc_subscribe = AsyncMock()
    bus.rpc_release_reply_channel = AsyncMock()
    codec = OrionCodec()

    async def _publish(channel: str, env: BaseEnvelope) -> None:
        reply = _env().model_copy(update={"correlation_id": env.correlation_id})
        await bus._handle_rpc_result({"channel": env.reply_to.encode(), "data": codec.encode(reply)})
        raise RuntimeError("publish failed after the reply landed")

    bus.publish = _publish  # type: ignore[assignment]
    req = BaseEnvelope(kind="k", source=SRC, reply_to="orion:r", payload={})
    with pytest.raises(RuntimeError):
        await bus.rpc_request("orion:bg", req, reply_channel="orion:r", timeout_sec=1.0)
    assert bus._rpc_reply_envelopes == {}
    bus._rpc_worker_task.cancel()


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


@pytest.mark.asyncio
async def test_rpc_request_inline_path_records_a_marked_reply_as_degraded() -> None:
    from unittest.mock import patch

    class _Redis:
        def __init__(self) -> None:
            self.pubsub_obj = _InlinePubSub()

        def pubsub(self):
            return self.pubsub_obj

        async def close(self) -> None:
            return None

    bus = OrionBusAsync("redis://unused:6379/0")
    fake = _Redis()
    codec = OrionCodec()
    reply = mark_degraded(_env(), "metacog_draft_fallback:empty_response")

    async def _publish(channel, env):
        await fake.pubsub_obj.queue.put({"type": "message", "data": codec.encode(reply)})

    bus.publish = _publish  # type: ignore[assignment]
    req = BaseEnvelope(kind="k", source=SRC, payload={})
    with patch.object(bus, "_create_pubsub_redis", return_value=fake):
        await bus.rpc_request("orion:bg", req, reply_channel="r1", timeout_sec=1.0, health_label="lbl")
    snap = bus.get_rpc_health_snapshot()
    hop = snap.channel_latency["orion:bg#lbl"]
    assert (hop.success_count, hop.degraded_count, hop.log_ms_sum) == (0, 1, 0.0)
    assert (snap.success_count, snap.degraded_count, snap.success_latency_ms_max) == (0, 1, None)


# --- contract: additive, old payloads still parse ---------------------------


def _old_snapshot_payload() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "service": "cortex-orch",
        "window_start": now,
        "window_end": now,
        "success_count": 3,
        "timeout_count": 0,
        "channel_counts": {"orion:a": 3},
        "channel_latency": {
            "orion:a": {"success_count": 3, "timeout_count": 0, "log_ms_sum": 6.0, "log_ms_sumsq": 12.5, "max_ms": 9.0}
        },
    }


def test_a_snapshot_from_a_producer_that_predates_degraded_count_still_parses() -> None:
    m = RpcHealthSnapshotV1.model_validate(_old_snapshot_payload())
    assert m.degraded_count == 0
    assert m.channel_latency["orion:a"].degraded_count == 0


def test_published_envelope_carries_degraded_count_and_validates() -> None:
    agg = RpcHealthAggregator()
    agg.record_success(request_channel="orion:a", latency_ms=5.0)
    agg.record_degraded(request_channel="orion:a")
    env = build_rpc_health_snapshot_envelope(
        agg.snapshot_and_reset(), service="s", node="n", instance=None, source=SRC, include_channel_latency=True
    )
    assert env.payload["degraded_count"] == 1
    assert env.payload["channel_latency"]["orion:a"]["degraded_count"] == 1
    RpcHealthSnapshotV1.model_validate(env.payload)
    with pytest.raises(Exception):
        RpcChannelLatencyV1.model_validate({"degraded_count": -1, "bogus": 1})


def test_registry_resolves_the_changed_models() -> None:
    from orion.schemas.registry import resolve

    assert resolve("RpcChannelLatencyV1") is RpcChannelLatencyV1
    assert resolve("RpcHealthSnapshotV1") is RpcHealthSnapshotV1
    assert "degraded_count" in RpcChannelLatencyV1.model_fields
