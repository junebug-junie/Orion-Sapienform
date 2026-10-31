"""RPCs that nobody hears fail fast instead of waiting out their timeout (2026-10-31).

Live: durable-runs published two curiosity turns at 00:52:26; the restarted Hub subscribed at
00:53:38. Pub/sub keeps nothing, so both requests were gone, and one run held a GPU lane for 35
minutes on a 2.5 h RPC timeout until it was cancelled by hand.

PUBLISH's own receiver count is no use here: it includes pattern subscribers, and orion-bus-mirror
PSUBSCRIBEs ``orion:*`` (verified against a real Redis: PUBLISH 1, NUMSUB 0 with only a pattern
subscriber). The bus reads PUBSUB NUMSUB in the same MULTI as the PUBLISH.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from orion.core.bus.async_service import OrionBusAsync, RpcNoListener
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

SRC = ServiceRef(name="test", version="0", node="athena")
CHANNEL = "orion:curiosity:turn:request"


class FakePipeline:
    def __init__(self, redis: "FakeRedis"):
        self.redis = redis

    def publish(self, channel, data):
        return self

    def pubsub_numsub(self, channel):
        return self

    async def execute(self):
        # PUBLISH counts the pattern-subscribing mirror too; NUMSUB only plain listeners.
        return [self.redis.at_publish + 1, [(CHANNEL.encode(), self.redis.at_publish)]]


class FakeRedis:
    def __init__(self, at_publish: int = 1, polls: list[int] | None = None, poll_error: bool = False):
        self.at_publish = at_publish
        self.polls = list(polls or [])
        self.poll_error = poll_error
        self.poll_calls = 0

    def pipeline(self, transaction=True):
        assert transaction, "PUBLISH and NUMSUB must be read in one MULTI"
        return FakePipeline(self)

    async def publish(self, channel, data):
        return 1   # the mirror always hears it

    async def pubsub_numsub(self, channel):
        self.poll_calls += 1
        if self.poll_error:
            raise ConnectionError("redis hiccup")
        return [(channel.encode(), self.polls.pop(0) if self.polls else 1)]


class Health:
    def __init__(self):
        self.timeouts = []

    def record_timeout(self, **kw):
        self.timeouts.append(kw)

    def __getattr__(self, name):   # record_success / record_degraded / hop records: not under test
        return lambda *a, **kw: None


def _bus(redis: FakeRedis, reply: asyncio.Future | None = None) -> OrionBusAsync:
    bus = OrionBusAsync("redis://test", enabled=True, enforce_catalog=False, track_velocity=False)
    bus._redis = redis
    bus._validate_payload = lambda channel, msg: None
    bus._rpc_health = Health()
    bus.grammar = []

    async def grammar(**kw):
        bus.grammar.append(kw)

    bus._emit_rpc_timeout_grammar = grammar

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
async def test_a_request_no_worker_received_fails_at_once_though_the_mirror_heard_it():
    bus = _bus(FakeRedis(at_publish=0))
    with pytest.raises(RpcNoListener, match="published to nobody"):
        await asyncio.wait_for(bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=3600,
                                               require_listener=True), timeout=2)
    assert len(bus._rpc_health.timeouts) == 1 and len(bus.grammar) == 1   # visible like a timeout


@pytest.mark.asyncio
async def test_without_the_flag_an_unheard_request_still_waits_as_before():
    bus = _bus(FakeRedis(at_publish=0))
    with pytest.raises(TimeoutError):
        await bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=0.2)


@pytest.mark.asyncio
async def test_a_listener_gone_on_two_checks_in_a_row_fails_the_wait():
    redis = FakeRedis(polls=[1, 0, 0])
    bus = _bus(redis)
    with pytest.raises(RpcNoListener, match="while awaiting"):
        await asyncio.wait_for(bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=3600,
                                               require_listener=True, listener_check_sec=0.05), timeout=2)
    assert redis.poll_calls == 3 and len(bus._rpc_health.timeouts) == 1


@pytest.mark.asyncio
async def test_a_listener_only_resubscribing_is_not_lost():
    """resilience.run_subscription drops and re-subscribes on errors; its request is still running."""
    loop = asyncio.get_running_loop()
    reply = loop.create_future()
    loop.call_later(0.4, reply.set_result, {"data": b"x"})
    redis = FakeRedis(polls=[0, 1, 0, 1, 0, 1, 0, 1])
    bus = _bus(redis, reply)
    msg = await bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=5,
                                require_listener=True, listener_check_sec=0.05)
    assert msg == {"data": b"x"} and bus._rpc_health.timeouts == []


@pytest.mark.asyncio
async def test_the_total_timeout_still_applies_and_a_failed_check_is_not_a_lost_listener():
    redis = FakeRedis(poll_error=True)
    bus = _bus(redis)
    with pytest.raises(TimeoutError):
        await bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=0.3,
                              require_listener=True, listener_check_sec=0.05)
    assert redis.poll_calls >= 2


@pytest.mark.asyncio
async def test_an_outer_cancel_still_cancels_the_caller():
    bus = _bus(FakeRedis())
    task = asyncio.ensure_future(bus.rpc_request(CHANNEL, _env(), reply_channel="r:1", timeout_sec=3600,
                                                 require_listener=True, listener_check_sec=0.05))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
