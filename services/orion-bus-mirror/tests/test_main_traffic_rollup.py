"""Mirror wiring for the 5-minute traffic rollup: message -> count, hop -> hop, publish fail-open."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.main as main_mod
from app.graph_writer import ChainTracker
from app.main import (
    _flush_traffic_rollup_once,
    _record_graph_event,
    _rollup_envelope,
    mirror_bus,
)
from app.rate_rollup import TrafficRollupAccumulator
from app.settings import settings
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import HeartbeatOnly
from orion.core.bus.codec import OrionCodec
from orion.schemas.bus_traffic import BUS_TRAFFIC_ROLLUP_CHANNEL, BUS_TRAFFIC_ROLLUP_KIND

T0 = 1_763_000_100


@dataclass
class _Src:
    name: Optional[str]
    node: Optional[str] = None


@dataclass
class _Env:
    source: Any
    correlation_id: Any = None
    payload: Any = None


@dataclass
class _Decoded:
    envelope: Any
    ok: bool = True


def test_rollup_setting_ships_on() -> None:
    assert type(settings).model_fields["MIRROR_TRAFFIC_ROLLUP_ENABLED"].default is True


@pytest.mark.asyncio
async def test_cross_organ_hop_feeds_rollup_without_a_graph_writer() -> None:
    rollup = TrafficRollupAccumulator(node="athena")
    tracker = ChainTracker(ttl_sec=60.0)
    first = _Decoded(_Env(source=_Src("cortex-exec"), correlation_id="c1"))
    second = _Decoded(_Env(source=_Src("llm-gateway"), correlation_id="c1"))

    await _record_graph_event(None, tracker, first, "orion:a", T0 + 1.0, set(), rollup)
    await _record_graph_event(None, tracker, second, "orion:b", T0 + 3.5, set(), rollup)

    [event] = rollup.flush_closed(T0 + 300)
    [hop] = event.hops
    assert (hop.from_organ, hop.to_organ, hop.count) == ("cortex-exec", "llm-gateway", 1)
    assert hop.latency_mean_sec == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_hop_still_feeds_rollup_when_graph_write_fails() -> None:
    writer = MagicMock()
    writer.record_publish.side_effect = RuntimeError("falkor down")
    rollup = TrafficRollupAccumulator(node="athena")
    tracker = ChainTracker(ttl_sec=60.0)
    await _record_graph_event(writer, tracker, _Decoded(_Env(_Src("a"), "c1")), "orion:a", T0, set(), rollup)
    await _record_graph_event(writer, tracker, _Decoded(_Env(_Src("b"), "c1")), "orion:b", T0 + 1, set(), rollup)
    [event] = rollup.flush_closed(T0 + 300)
    assert [(h.from_organ, h.to_organ) for h in event.hops] == [("a", "b")]


@pytest.mark.asyncio
async def test_same_organ_repeat_is_not_a_hop() -> None:
    rollup = TrafficRollupAccumulator(node="athena")
    tracker = ChainTracker(ttl_sec=60.0)
    for t in (0.0, 1.0):
        await _record_graph_event(None, tracker, _Decoded(_Env(_Src("a"), "c1")), "orion:a", T0 + t, set(), rollup)
    assert rollup.flush_closed(T0 + 300) == []


def test_rollup_envelope_passes_the_bus_catalog_validation() -> None:
    rollup = TrafficRollupAccumulator(node="athena")
    rollup.observe_message("orion:a", T0)
    rollup.observe_hop("a", "b", 1.0, T0)
    [event] = rollup.flush_closed(T0 + 300)
    env = _rollup_envelope(event)
    assert env.kind == BUS_TRAFFIC_ROLLUP_KIND
    # The real publish-side check: channel is cataloged and the payload matches its schema_id.
    OrionBusAsync(url="redis://unused:6379/0", enabled=True)._validate_payload(BUS_TRAFFIC_ROLLUP_CHANNEL, env)


@pytest.mark.asyncio
async def test_flush_publishes_closed_buckets_in_order() -> None:
    rollup = TrafficRollupAccumulator(node="athena")
    rollup.observe_message("orion:a", T0)
    rollup.observe_message("orion:a", T0 + 300)
    bus = MagicMock()
    bus.publish = AsyncMock()
    pending: deque = deque(maxlen=12)

    assert await _flush_traffic_rollup_once(bus, rollup, pending, T0 + 600) == 2
    channels = [c.args[0] for c in bus.publish.await_args_list]
    starts = [c.args[1].payload["bucket_start"] for c in bus.publish.await_args_list]
    assert channels == [BUS_TRAFFIC_ROLLUP_CHANNEL] * 2
    assert starts == sorted(starts)
    assert not pending


@pytest.mark.asyncio
async def test_publish_failure_is_fail_open_and_retried_next_tick() -> None:
    rollup = TrafficRollupAccumulator(node="athena")
    rollup.observe_message("orion:a", T0)
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=ConnectionError("bus down"))
    pending: deque = deque(maxlen=12)

    assert await _flush_traffic_rollup_once(bus, rollup, pending, T0 + 300) == 0  # does not raise
    assert len(pending) == 1

    bus.publish = AsyncMock()
    assert await _flush_traffic_rollup_once(bus, rollup, pending, T0 + 300) == 1
    assert not pending


@pytest.mark.asyncio
async def test_pending_queue_is_bounded_during_a_long_outage() -> None:
    rollup = TrafficRollupAccumulator(node="athena")
    for i in range(5):
        rollup.observe_message("orion:a", T0 + 300 * i)
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=ConnectionError("bus down"))
    pending: deque = deque(maxlen=3)
    await _flush_traffic_rollup_once(bus, rollup, pending, T0 + 300 * 5)
    assert len(pending) == 3
    assert pending[0].bucket_start.timestamp() == T0 + 600  # oldest two dropped


class _Session:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = list(messages)

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(0.01)
        return None


class _FakeBus:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages
        self.codec = OrionCodec()
        self.connect = AsyncMock()
        self.close = AsyncMock()
        self.publish = AsyncMock()

    def subscribe(self, *channels: str, patterns: bool = False):
        from contextlib import asynccontextmanager

        bus = self

        @asynccontextmanager
        async def _cm():
            yield _Session(bus._messages)

        return _cm()


@pytest.mark.asyncio
async def test_mirror_counts_messages_and_hops_and_publishes_the_rollup(tmp_path, monkeypatch) -> None:
    """End to end through mirror_bus(), graph OFF: two organs share a correlation_id,
    plus one undecodable message. The published bucket counts all three and the hop."""
    codec = OrionCodec()
    a = BaseEnvelope(kind="test.event.v1", source=ServiceRef(name="cortex-exec"), correlation_id="7d6c1f0e-0000-4000-8000-000000000001", payload={})
    b = BaseEnvelope(kind="test.event.v1", source=ServiceRef(name="llm-gateway"), correlation_id="7d6c1f0e-0000-4000-8000-000000000001", payload={})
    bus = _FakeBus([
        {"type": "pmessage", "channel": b"orion:test:a", "data": codec.encode(a)},
        {"type": "pmessage", "channel": b"orion:test:b", "data": codec.encode(b)},
        {"type": "pmessage", "channel": b"orion:test:a", "data": b"not json"},
    ])
    monkeypatch.setattr(settings, "MIRROR_PARQUET_DIR", str(tmp_path / "parquet"))
    monkeypatch.setattr(settings, "MIRROR_SQLITE_PATH", str(tmp_path / "mirror.sqlite"))
    monkeypatch.setattr(settings, "MIRROR_GRAPH_ENABLED", False)
    monkeypatch.setattr(settings, "MIRROR_TRAFFIC_ROLLUP_ENABLED", True)
    monkeypatch.setattr(main_mod, "TRAFFIC_ROLLUP_FLUSH_CHECK_SEC", 0.02)

    built: list[TrafficRollupAccumulator] = []

    def _factory(**kw):
        # 1-second buckets, no startup drop, so the test closes a bucket quickly.
        kw.update(bucket_sec=1, started_at_epoch=None)
        acc = TrafficRollupAccumulator(**kw)
        built.append(acc)
        return acc

    monkeypatch.setattr(main_mod, "TrafficRollupAccumulator", _factory)

    def _published() -> list[dict]:
        return [c.args[1].payload for c in bus.publish.await_args_list if c.args[0] == BUS_TRAFFIC_ROLLUP_CHANNEL]

    with patch("app.main.OrionBusAsync", return_value=bus), patch(
        "app.main.build_heartbeat_chassis", return_value=AsyncMock(spec=HeartbeatOnly)
    ):
        task = asyncio.create_task(mirror_bus())
        loop = asyncio.get_running_loop()
        end = loop.time() + 5
        while sum(p["message_count"] for p in _published()) < 3:
            assert loop.time() < end, "rollup never published"
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    payloads = _published()
    assert len(built) == 1
    assert sum(p["message_count"] for p in payloads) == 3
    assert sum(p["decode_failures"] for p in payloads) == 1
    totals: dict[str, int] = {}
    for p in payloads:
        for k, v in p["channel_counts"].items():
            totals[k] = totals.get(k, 0) + v
    assert totals == {"orion:test:a": 2, "orion:test:b": 1}
    hops = [h for p in payloads for h in p["hops"]]
    assert [(h["from_organ"], h["to_organ"], h["count"]) for h in hops] == [("cortex-exec", "llm-gateway", 1)]
