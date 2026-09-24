"""The real client (orion.gpu_pool.client) against a real in-process pool, through a fake bus that
uses the real OrionCodec both ways and hands back raw pubsub-shaped messages exactly like
rpc_request does -- so an envelope/codec mismatch fails here instead of passing for the wrong reason."""
from __future__ import annotations

import asyncio
import contextlib
import uuid

import pytest

import app.main as main_mod
from app.runtime import PoolRuntime
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.gpu_pool.client import LeaseBacklogged, LeaseUnavailable, gpu_lease
from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL, GPU_POOL_LEASE_REQUEST_CHANNEL

from tests.test_runtime import boot, make


class WiredBus:
    """Client side and pool side on one in-memory pubsub."""

    def __init__(self):
        self.codec = OrionCodec()
        self.subscribers: dict[str, list[asyncio.Queue]] = {}
        self.rpc_labels: list[str | None] = []
        self.turn_corrs: set[str] = set()
        self.hops: list = []

    # pool side (PoolRuntime.bus)
    async def publish(self, channel, env):
        for q in self.subscribers.get(channel, []):
            q.put_nowait({"type": "message", "channel": channel, "data": self.codec.encode(env)})

    def record_hop_success(self, hop, ms):
        self.hops.append(hop)

    def record_hop_timeout(self, hop, ms=None):
        self.hops.append(hop)

    # client side
    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec=60.0, health_label=None):
        assert request_channel == GPU_POOL_LEASE_REQUEST_CHANNEL
        self.rpc_labels.append(health_label)
        self.turn_corrs.add(str(envelope.correlation_id))
        wire = self.codec.decode(self.codec.encode(envelope)).envelope
        reply = await main_mod._on_lease(wire)
        return {"type": "message", "channel": reply_channel, "data": self.codec.encode(reply)}

    @contextlib.asynccontextmanager
    async def subscribe(self, *channels, patterns=False):
        q: asyncio.Queue = asyncio.Queue()
        for c in channels:
            self.subscribers.setdefault(c, []).append(q)
        try:
            yield q
        finally:
            for c in channels:
                self.subscribers[c].remove(q)

    async def iter_messages(self, q):
        while True:
            yield await q.get()


async def pool(bus, **kw):
    rt, clock = make(bus=bus, **kw)
    await boot(rt)
    main_mod.runtime = rt
    return rt, clock


def test_granted_lease_carries_the_discovered_model_and_releases_ok():
    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        turn = str(uuid.uuid4())
        async with gpu_lease(bus, work_class="metacog", holder="t", turn_correlation_id=turn) as lease:
            assert lease.grant.role == "metacog" and lease.grant.model_file == "Qwen_Qwen3-8B-Q5_K_M.gguf"
            lease_id = lease.lease_id
        assert (await rt.store.lease(lease_id))["status"] == "released"
        assert (await rt.store.lease(lease_id))["turn_correlation_id"] == turn
        assert turn not in bus.turn_corrs          # the lease RPC never rides the turn's corr id
        assert set(bus.rpc_labels) == {"gpu_pool_lease"}
    asyncio.run(go())


def test_queued_caller_is_woken_by_the_grant_event():
    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        holder_done = asyncio.Event()

        async def first():
            async with gpu_lease(bus, work_class="chat", holder="a", priority="interactive"):
                await holder_done.wait()

        task = asyncio.create_task(first())
        await asyncio.sleep(0.05)

        async def second():
            async with gpu_lease(bus, work_class="chat", holder="b", priority="interactive", deadline_sec=5) as lease:
                return lease.grant.role

        waiter = asyncio.create_task(second())
        await asyncio.sleep(0.05)
        assert not waiter.done()
        holder_done.set()
        assert await asyncio.wait_for(waiter, 5) == "chat"
        await task
    asyncio.run(go())


def test_exception_in_block_releases_as_upstream_error_and_retries():
    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        with pytest.raises(RuntimeError):
            async with gpu_lease(bus, work_class="fast", holder="t") as lease:
                lease_id = lease.lease_id
                raise RuntimeError("HTTP 500 from llama.cpp")
        row = await rt.store.lease(lease_id)
        assert row["status"] == "retry_wait" and "HTTP 500" in row["reason"]
    asyncio.run(go())


def test_backlogged_and_unavailable_are_typed_not_timeouts():
    async def go():
        bus = WiredBus()
        await pool(bus, down=("world",))
        with pytest.raises(LeaseBacklogged):
            async with gpu_lease(bus, work_class="world", holder="t"):
                pass
        with pytest.raises(LeaseUnavailable) as err:
            async with gpu_lease(bus, work_class="experiment", holder="t"):
                pass
        assert err.value.reason == "operator_only_class"
    asyncio.run(go())


def test_deadline_while_queued_cancels_so_nothing_is_left_holding_a_slot():
    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        release = asyncio.Event()

        async def hold():
            async with gpu_lease(bus, work_class="chat", holder="a", priority="interactive"):
                await release.wait()

        task = asyncio.create_task(hold())
        await asyncio.sleep(0.05)
        with pytest.raises(LeaseUnavailable):
            async with gpu_lease(bus, work_class="chat", holder="b", priority="interactive", deadline_sec=0.1):
                pass
        cancelled = [r for r in rt.store.leases.values() if r["holder"] == "b"]
        assert cancelled and cancelled[0]["status"] == "released"
        release.set()
        await task
    asyncio.run(go())
