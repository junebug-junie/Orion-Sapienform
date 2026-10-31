"""RPCs that nobody hears fail fast instead of waiting out their timeout (2026-10-31).

Live: durable-runs published two curiosity turns at 00:52:26; the restarted Hub subscribed at
00:53:38. Pub/sub keeps nothing, so both requests were gone, and one run held a GPU lane for 35
minutes on a 2.5 h RPC timeout until it was cancelled by hand.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from orion.core.bus.async_service import OrionBusAsync, RpcNoListener
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

SRC = ServiceRef(name="test", version="0", node="athena")
CHANNEL = "orion:curiosity:turn:request"


class FakeRedis:
    """PUBLISH answers with the receiver count; PUBSUB NUMSUB with the current subscribers."""

    def __init__(self, receivers: int = 1, numsub: list[int] | None = None, numsub_error: bool = False):
        self.receivers = receivers
        self.numsub = list(numsub or [])
        self.numsub_error = numsub_error
        self.numsub_calls = 0

    async def publish(self, channel, data):
        return self.receivers

    async def pubsub_numsub(self, channel):
        self.numsub_calls += 1
        if self.numsub_error:
            raise ConnectionError("redis hiccup")
        n = self.numsub.pop(0) if self.numsub else 1
        return [(channel.encode(), n)]


def _bus(redis: FakeRedis, reply: asyncio.Future | None = None) -> OrionBusAsync:
    bus = OrionBusAsync("redis://test", enabled=True, enforce_catalog=False, track_velocity=False)
    bus._redis = redis
    bus._validate_payload = lambda channel, msg: None

    @asynccontextmanager
    async def subscribe(*channels, patterns=False):
        yield object()

    async def iter_messages(pubsub):
        if reply is not None:
            yield await reply
        else:
            await asyncio.sleep(3600)
            yield {}

    bus.subscribe = subscribe
    bus.iter_messages = iter_messages
    return bus


def _env() -> BaseEnvelope:
    return BaseEnvelope(kind="curiosity.turn.request", source=SRC, payload={})


@pytest.mark.asyncio
async def test_publish_returns_how_many_subscribers_received_it():
    bus = _bus(FakeRedis(receivers=2))
    assert await bus.publish(CHANNEL, _env()) == 2
    bus.enabled = False
    assert await bus.publish(CHANNEL, _env()) is None


@pytest.mark.asyncio
async def test_a_request_nobody_received_fails_at_once():
    bus = _bus(FakeRedis(receivers=0))
    with pytest.raises(RpcNoListener, match="published to nobody"):
        await asyncio.wait_for(bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=3600,
                                               require_listener=True), timeout=2)


@pytest.mark.asyncio
async def test_without_the_flag_an_unheard_request_still_waits_as_before():
    bus = _bus(FakeRedis(receivers=0))
    with pytest.raises(TimeoutError):
        await bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=0.2)


@pytest.mark.asyncio
async def test_a_listener_that_vanishes_mid_wait_fails_within_one_check():
    redis = FakeRedis(receivers=1, numsub=[1, 0])
    bus = _bus(redis)
    with pytest.raises(RpcNoListener, match="while awaiting"):
        await asyncio.wait_for(bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=3600,
                                               require_listener=True, listener_check_sec=0.05), timeout=2)
    assert redis.numsub_calls == 2


@pytest.mark.asyncio
async def test_a_listener_that_stays_gets_its_reply():
    loop = asyncio.get_running_loop()
    reply = loop.create_future()
    loop.call_later(0.2, reply.set_result, {"data": b"x"})
    bus = _bus(FakeRedis(receivers=1), reply)
    msg = await bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=5,
                                require_listener=True, listener_check_sec=0.05)
    assert msg == {"data": b"x"}


@pytest.mark.asyncio
async def test_the_total_timeout_still_applies_and_a_failed_check_is_not_a_lost_listener():
    redis = FakeRedis(receivers=1, numsub_error=True)
    bus = _bus(redis)
    with pytest.raises(TimeoutError):
        await bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=0.3,
                              require_listener=True, listener_check_sec=0.05)
    assert redis.numsub_calls >= 2
