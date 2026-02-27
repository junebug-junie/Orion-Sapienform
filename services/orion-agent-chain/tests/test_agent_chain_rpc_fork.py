from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


class _FakePubSub:
    def __init__(self):
        self.channels = set()
        self.messages = asyncio.Queue()

    async def subscribe(self, *channels):
        for c in channels:
            self.channels.add(c)

    async def unsubscribe(self, *channels):
        for c in channels:
            self.channels.discard(c)

    async def close(self):
        return None

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        try:
            return await asyncio.wait_for(self.messages.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class _FakeRedis:
    def __init__(self):
        self.pubsubs = []

    async def ping(self):
        return True

    def pubsub(self):
        ps = _FakePubSub()
        self.pubsubs.append(ps)
        return ps

    async def publish(self, channel, data):
        # broadcast to all subscribed pubsubs
        for ps in self.pubsubs:
            if channel in ps.channels:
                await ps.messages.put({"type": "message", "channel": channel.encode("utf-8"), "data": data})

    async def close(self):
        return None


class _FakeCodec:
    def encode(self, msg):
        return msg

    def decode(self, data):
        return SimpleNamespace(ok=True, envelope=data, error=None)


async def _run():
    bus = OrionBusAsync("redis://fake", codec=_FakeCodec(), enforce_catalog=False)
    fake_redis = _FakeRedis()
    bus._redis = fake_redis

    original_connect = OrionBusAsync.connect

    async def _fake_connect(self):
        if self._redis is None:
            self._redis = fake_redis
        return None

    OrionBusAsync.connect = _fake_connect
    try:
        child = await bus.fork(start_rpc_worker=True)
    finally:
        OrionBusAsync.connect = original_connect

    child_corr = str(uuid4())
    reply_ch = f"orion:exec:result:PlannerReactService:{child_corr}"
    env = BaseEnvelope(
        kind="agent.planner.request",
        source=ServiceRef(name="test", version="1", node="n"),
        correlation_id=child_corr,
        reply_to=reply_ch,
        payload={"x": 1},
    )

    async def _reply_later():
        await asyncio.sleep(0.01)
        reply = BaseEnvelope(
            kind="agent.planner.result",
            source=ServiceRef(name="planner", version="1", node="n"),
            correlation_id=child_corr,
            payload={"status": "ok"},
        )
        await child.publish(reply_ch, reply)

    t = asyncio.create_task(_reply_later())
    msg = await child.rpc_request(
        "orion:exec:request:PlannerReactService",
        env,
        reply_channel=reply_ch,
        timeout_sec=1,
    )
    await t
    decoded = child.codec.decode(msg.get("data"))
    assert decoded.envelope.payload.get("status") == "ok"
    await child.close()


def test_rpc_fork_worker_resolves_future():
    asyncio.run(_run())
