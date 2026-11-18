from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_STEP_CHANNEL,
    REVERIE_VISUAL_STEP_REQUEST_KIND,
    REVERIE_VISUAL_STEP_RESULT_KIND,
    ReverieVisualStepRequestV1,
    ReverieVisualStepResultV1,
)


def _bus_for(envelope):
    class _Decoded:
        ok = True
        error = None

    _Decoded.envelope = envelope

    class _Codec:
        @staticmethod
        def decode(data):
            return _Decoded()

    bus = AsyncMock()
    bus.codec = _Codec()
    return bus


def _step_envelope(payload, *, kind=REVERIE_VISUAL_STEP_REQUEST_KIND):
    corr = uuid4()
    return BaseEnvelope(
        kind=kind, source=ServiceRef(name="orion-durable-runs"), correlation_id=corr,
        reply_to=f"orion:reverie:visual:step:reply:{corr}", causality_chain=[],
        payload=payload,
    )


@pytest.mark.asyncio
async def test_step_channel_routes_to_visual_steps_and_replies_under_request_correlation():
    from app import bus_listener

    request = ReverieVisualStepRequestV1(
        run_id="run-1", correlation_id="step-corr-1", step="prepare",
        visual_request=VisualRunRequestV1(dispatch_id="dispatch-1"),
    )
    envelope = _step_envelope(request.model_dump(mode="json"), kind="some.other.kind")
    bus = _bus_for(envelope)
    result = ReverieVisualStepResultV1(run_id="run-1", correlation_id="step-corr-1", step="prepare",
                                       status="done", attempt_id="attempt-1", elapsed_sec=1.5)
    runner = AsyncMock(return_value=result)
    with patch.object(bus_listener, "run_visual_step", runner), \
            patch.object(bus_listener, "run_stance_react", AsyncMock(side_effect=AssertionError("stance"))):
        await bus_listener._handle_bus_message(
            bus, {"channel": REVERIE_VISUAL_STEP_CHANNEL.encode(), "data": b"ignored"})

    assert runner.await_args.args[1] == request
    bus.publish.assert_awaited_once()
    channel, reply = bus.publish.await_args.args
    assert channel == envelope.reply_to
    assert reply.kind == REVERIE_VISUAL_STEP_RESULT_KIND
    assert reply.correlation_id == envelope.correlation_id
    parsed = ReverieVisualStepResultV1.model_validate(reply.payload)
    assert (parsed.correlation_id, parsed.status, parsed.attempt_id) == ("step-corr-1", "done", "attempt-1")
    # A waking reply carries no dream-only `caption: null` (nor any other None field).
    assert "caption" not in reply.payload and None not in reply.payload.values()


@pytest.mark.asyncio
async def test_step_kind_routes_even_without_channel_metadata():
    from app import bus_listener

    request = ReverieVisualStepRequestV1(
        run_id="run-1", correlation_id="c", step="abandon", attempt_id="a-1",
        visual_request=VisualRunRequestV1(dispatch_id="dispatch-1"),
    )
    bus = _bus_for(_step_envelope(request.model_dump(mode="json")))
    result = ReverieVisualStepResultV1(run_id="run-1", correlation_id="c", step="abandon", status="done",
                                       attempt_id="a-1")
    with patch.object(bus_listener, "run_visual_step", AsyncMock(return_value=result)) as runner:
        await bus_listener._handle_bus_message(bus, {"data": b"ignored"})
    runner.assert_awaited_once()
    assert bus.publish.await_args.args[1].kind == REVERIE_VISUAL_STEP_RESULT_KIND


@pytest.mark.asyncio
async def test_invalid_step_request_is_a_retry_never_terminal():
    # Schema skew during a rolling deploy must not kill an in-flight run.
    from app import bus_listener

    envelope = _step_envelope({"run_id": "run-1", "correlation_id": "c", "step": "generate",
                               "visual_request": {}})  # generate without attempt_id/gpu_lease
    bus = _bus_for(envelope)
    with patch.object(bus_listener, "run_visual_step", AsyncMock(side_effect=AssertionError("ran"))):
        await bus_listener._handle_bus_message(bus, {"data": b"ignored"})
    reply = bus.publish.await_args.args[1]
    assert reply.correlation_id == envelope.correlation_id
    parsed = ReverieVisualStepResultV1.model_validate(reply.payload)
    assert (parsed.status, parsed.outcome, parsed.reason) == ("retry", None, "invalid_step_request")
    assert parsed.retry_after_sec and parsed.retry_after_sec > 0


@pytest.mark.asyncio
async def test_invalid_step_request_without_ids_still_answers_under_envelope_correlation():
    from app import bus_listener

    envelope = _step_envelope({"step": "prepare"})
    bus = _bus_for(envelope)
    await bus_listener._handle_bus_message(bus, {"data": b"ignored"})
    reply = bus.publish.await_args.args[1]
    assert reply.correlation_id == envelope.correlation_id
    assert reply.payload["correlation_id"] == str(envelope.correlation_id)
    assert reply.payload["reason"] == "invalid_step_request"


@pytest.mark.asyncio
async def test_missing_subscriptions_probes_every_channel():
    from app import bus_listener

    counts = {"orion:thought:request": 1, REVERIE_VISUAL_STEP_CHANNEL: 0, "probe-fails": None}

    async def numsub(channel):
        if counts[channel] is None:
            raise ConnectionError("redis down")
        return [(channel.encode(), counts[channel])]

    bus = AsyncMock()
    bus.redis.pubsub_numsub = numsub
    missing = await bus_listener._missing_subscriptions(bus, tuple(counts))
    assert missing == [REVERIE_VISUAL_STEP_CHANNEL]


@pytest.mark.asyncio
async def test_worker_subscribes_both_channels_and_reconnects_when_step_channel_drops(monkeypatch):
    from app import bus_listener

    stop = asyncio.Event()
    subscribed: list[tuple[str, ...]] = []

    class _PubSub:
        async def get_message(self, **kw):
            raise asyncio.TimeoutError

    class _FakeBus:
        def __init__(self, url):
            self.redis = AsyncMock()
            self.redis.pubsub_numsub = AsyncMock(side_effect=lambda ch: [
                (ch, 0 if ch == REVERIE_VISUAL_STEP_CHANNEL else 1)])

        async def connect(self):
            return None

        @asynccontextmanager
        async def subscribe(self, *channels):
            subscribed.append(channels)
            yield _PubSub()

        async def close(self):
            stop.set()

    monkeypatch.setattr(bus_listener.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(bus_listener, "OrionBusAsync", _FakeBus)
    monkeypatch.setattr(bus_listener, "_PUBSUB_IDLE_POLLS_BEFORE_HEALTH", 1)
    await asyncio.wait_for(bus_listener.run_bus_worker(stop), timeout=5)
    assert subscribed == [(bus_listener.settings.channel_thought_request, REVERIE_VISUAL_STEP_CHANNEL)]
    assert stop.is_set()
