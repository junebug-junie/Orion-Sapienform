"""#3920: every pool event also lands on the durable stream sql-writer reads through a consumer group.

Pub/sub keeps nothing, so events published while sql-writer was rebuilding never reached
gpu_pool_events (68 of 69 missing grants on 2026-11-24/25 fell inside its rebuild windows).
These pin the producer half: the stream copy is the same envelope as the pub/sub one, is written
even when the publish fails, and a stream failure never breaks scheduling.
"""
from __future__ import annotations

import fakeredis.aioredis

from orion.core.bus.work_queue import RedisStreamWorkQueue
from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND, GPU_POOL_EVENT_STREAM
from tests.test_runtime import FakeBus, acq, boot, make, run


class Sink:
    def __init__(self, fail=False):
        self.envs, self.fail = [], fail

    async def __call__(self, env):
        if self.fail:
            raise ConnectionError("redis down")
        self.envs.append(env)


def test_every_published_pool_event_is_also_streamed_as_the_same_envelope():
    async def go():
        rt, _ = make()
        rt.event_stream = sink = Sink()
        await boot(rt)
        r = await rt.acquire(acq("metacog"))
        await rt.release(r.lease_id, "ok")
        published = [e for c, e in rt.bus.published if c == GPU_POOL_EVENT_CHANNEL]
        assert published and [e.payload["event_id"] for e in published] == [e.payload["event_id"] for e in sink.envs]
        assert all(s is p for s, p in zip(sink.envs, published))  # one envelope, two transports
        assert {e.kind for e in sink.envs} == {GPU_POOL_EVENT_KIND}
        assert {"admitted", "granted", "released"} <= {e.payload["event"] for e in sink.envs
                                                       if e.payload["lease_id"] == r.lease_id}
    run(go())


def test_a_failed_publish_still_streams_and_a_failed_stream_never_breaks_a_grant():
    class DownBus(FakeBus):
        async def publish(self, channel, env):
            raise ConnectionError("pubsub down")

    async def go():
        rt, _ = make(bus=DownBus())
        rt.event_stream = sink = Sink()
        await boot(rt)
        r = await rt.acquire(acq("metacog"))
        assert r.status == "granted"
        assert any(e.payload["event"] == "granted" and e.payload["lease_id"] == r.lease_id for e in sink.envs)

        rt2, _ = make()
        rt2.event_stream = Sink(fail=True)
        await boot(rt2)
        r2 = await rt2.acquire(acq("metacog"))
        assert r2.status == "granted"
        assert rt2.bus.events("granted")  # pub/sub unaffected
    run(go())


def test_the_stream_entry_decodes_back_to_the_event_for_a_consumer_group():
    """The real sink main.py builds: RedisStreamWorkQueue.enqueue onto GPU_POOL_EVENT_STREAM."""
    async def go():
        queue = RedisStreamWorkQueue("redis://unused", client=fakeredis.aioredis.FakeRedis(), default_maxlen=100)
        await queue.ensure_group(GPU_POOL_EVENT_STREAM, "cg:test", start_id="0")
        rt, _ = make()

        async def sink(env):
            await queue.enqueue(GPU_POOL_EVENT_STREAM, env)

        rt.event_stream = sink
        await boot(rt)
        r = await rt.acquire(acq("metacog"))
        batch = await queue.read_group(GPU_POOL_EVENT_STREAM, "cg:test", "c1", count=100, block_ms=10)
        got = [m.envelope.payload for m in batch.messages]
        assert not batch.decode_errors
        assert any(p["event"] == "granted" and p["lease_id"] == r.lease_id for p in got)
        assert [p["event_id"] for p in got] == [e.payload["event_id"] for c, e in rt.bus.published
                                                if c == GPU_POOL_EVENT_CHANNEL]
    run(go())
