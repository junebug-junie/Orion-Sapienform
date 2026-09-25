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
        # Like the real Rabbit chassis: no reply_to on the envelope means no reply, ever.
        # (This fake used to answer regardless, which hid a client that never set reply_to.)
        if envelope.reply_to != reply_channel:
            raise asyncio.TimeoutError(f"responder would reply to {envelope.reply_to!r}, caller listens on {reply_channel!r}")
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


def test_exception_in_block_releases_as_upstream_error():
    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        with pytest.raises(RuntimeError):
            async with gpu_lease(bus, work_class="fast", holder="t") as lease:
                lease_id = lease.lease_id
                raise RuntimeError("HTTP 500 from llama.cpp")
        row = await rt.store.lease(lease_id)
        assert row["status"] == "released" and "HTTP 500" in row["reason"]   # not retryable: ends
    asyncio.run(go())


def test_backlogged_and_unavailable_are_typed_not_timeouts():
    async def go():
        bus = WiredBus()
        await pool(bus, down=("world",))
        with pytest.raises(LeaseBacklogged):                  # opted in: the pool keeps it
            async with gpu_lease(bus, work_class="world", holder="t", retryable=True):
                pass
        with pytest.raises(LeaseUnavailable) as waited:       # default: waits, then a typed deadline
            async with gpu_lease(bus, work_class="world", holder="t", deadline_sec=0.2):
                pass
        assert waited.value.reason == "deadline" and not isinstance(waited.value, LeaseBacklogged)
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


def test_release_outcome_set_by_the_caller_reaches_the_pool():
    """The gateway stops an upstream call when the pool recalls/loses the lease and sets
    ``release_outcome="cancelled"``: that, not ok/upstream_error, is what the pool records."""
    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        sent = []
        real = main_mod._on_lease

        async def spy(env):
            sent.append(env.payload)
            return await real(env)

        main_mod._on_lease = spy
        try:
            async with gpu_lease(bus, work_class="fast", holder="t") as lease:
                lease.release_outcome, lease.release_detail = "cancelled", "gpu_pool_recalled:lost"
        finally:
            main_mod._on_lease = real
        release = [p for p in sent if p.get("verb") == "release"][-1]
        assert release["outcome"] == "cancelled" and release["detail"] == "gpu_pool_recalled:lost"
    asyncio.run(go())


class SilentBus(WiredBus):
    """A pool that never answers: every lease RPC times out after its own timeout_sec."""

    def __init__(self):
        super().__init__()
        self.rpcs: list[tuple[str, float]] = []

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec=60.0, health_label=None):
        self.rpcs.append((envelope.payload.get("verb"), timeout_sec))
        raise asyncio.TimeoutError()


@pytest.mark.parametrize("deadline_sec, expected", [(60.0, 10.0), (3.0, 3.0), (0.2, 1.0)])
def test_lease_rpc_timeout_is_bounded_by_the_deadline_and_withdraws_in_the_background(deadline_sec, expected):
    async def go():
        bus = SilentBus()
        with pytest.raises(asyncio.TimeoutError) as err:
            async with gpu_lease(bus, work_class="fast", holder="t", deadline_sec=deadline_sec):
                pass
        # The caller waited exactly one RPC; only a full-length wait may be read as "pool down".
        assert bus.rpcs == [("acquire", expected)]
        assert err.value.full is (expected == 10.0)
        # A slow (not dead) pool may still have admitted it: a bounded withdraw follows in the background.
        for _ in range(5):
            await asyncio.sleep(0)
        assert bus.rpcs[1:] == [("acquire", 2.0)]
    asyncio.run(go())


def test_a_refusal_while_queued_carries_the_pools_reason():
    """The pool refused a queued lease (e.g. the only big-enough role went away): the caller sees
    that reason, not a generic "deadline"."""
    from datetime import datetime, timedelta, timezone

    from orion.gpu_pool.client import _wait_for_grant

    async def go():
        bus = WiredBus()
        async with bus.subscribe(GPU_POOL_EVENT_CHANNEL) as q:
            env = BaseEnvelope(kind="gpu_pool.event.v1", source=ServiceRef(name="orion-gpu-pool"),
                               payload={"lease_id": "L1", "event": "unavailable",
                                        "reason": "min_ctx_exceeds_class:65536"})
            await bus.publish(GPU_POOL_EVENT_CHANNEL, env)
            lease, reason = await _wait_for_grant(bus, q, "L1", datetime.now(timezone.utc) + timedelta(seconds=1))
        assert lease is None and reason == "min_ctx_exceeds_class:65536"
    asyncio.run(go())


def test_withdraw_rpcs_are_bounded_short():
    """Cancelled while queued with the acquire reply in hand: the cancel RPC waits at most 2s."""
    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        timeouts = []
        real = bus.rpc_request

        async def spy(request_channel, envelope, *, reply_channel, timeout_sec=60.0, health_label=None):
            timeouts.append((envelope.payload.get("verb"), timeout_sec))
            return await real(request_channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec,
                              health_label=health_label)

        bus.rpc_request = spy
        release = asyncio.Event()

        async def hold():
            async with gpu_lease(bus, work_class="chat", holder="a", priority="interactive"):
                await release.wait()

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0.05)
        with pytest.raises(LeaseUnavailable):
            async with gpu_lease(bus, work_class="chat", holder="b", priority="interactive", deadline_sec=0.1):
                pass
        assert ("cancel", 2.0) in timeouts
        release.set()
        await holder
    asyncio.run(go())


# --- stage 4.3: the durable-run hold API (what 4.4 gateway / 4.5 durable-runs call) ---------------
def test_hold_lifecycle_through_the_real_client_and_codec():
    from orion.gpu_pool.client import (
        acquire_hold, durable_run_holder, gpu_lease, heartbeat_lease, hold_ref, lease_status, release_lease,
        validate_hold_ref,
    )

    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        assert durable_run_holder("run-1") == "durable-runs:run-1"
        reply = await acquire_hold(bus, holder="durable-runs:run-1", work_class="agent", request_id="run-1:1")
        assert reply.status == "granted" and reply.grant.role == "agent"
        again = await acquire_hold(bus, holder="durable-runs:run-1", work_class="agent", request_id="run-1:1")
        assert again.lease_id == reply.lease_id                   # idempotent on request_id
        ref = hold_ref(reply, "durable-runs:run-1")
        assert (ref.lease_id, ref.generation, ref.role) == (reply.lease_id, reply.grant.generation, "agent")
        await validate_hold_ref(bus, ref, source="hub", expected_holder="durable-runs:run-1")   # 4.4's fence
        async with gpu_lease(bus, work_class="agent", holder="orion-llm-gateway", hold=ref,
                             turn_correlation_id=str(uuid.uuid4())) as call:
            assert call.grant.role == "agent" and call.lease_id != ref.lease_id
            child = call.lease_id
        row = await rt.store.lease(child)
        assert row["status"] == "released" and row["hold_lease_id"] == ref.lease_id
        assert (await heartbeat_lease(bus, ref.lease_id, source="durable-runs")).status == "granted"
        st = await lease_status(bus, ref.lease_id, source="durable-runs")
        assert st.status == "granted" and st.grant.generation == ref.generation
        assert (await lease_status(bus, "missing", source="durable-runs")).status == "unknown_lease"
        assert (await release_lease(bus, ref.lease_id, source="durable-runs")).status == "ok"
        with pytest.raises(LeaseUnavailable) as err:
            async with gpu_lease(bus, work_class="agent", holder="orion-llm-gateway", hold=ref):
                pass
        assert err.value.reason == "hold_not_granted:released"
        with pytest.raises(LeaseUnavailable):
            await validate_hold_ref(bus, ref, source="hub")
    asyncio.run(go())


def test_a_refused_attach_never_cancels_the_runs_hold():
    """The client withdraws (cancels) the reply's lease_id when an acquire/attach fails. A refused
    attach must therefore never name the hold there, or one stale call would end the whole run."""
    from orion.gpu_pool.client import acquire_hold, gpu_lease, hold_ref, validate_hold_ref
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    async def go():
        bus = WiredBus()
        rt, _ = await pool(bus)
        reply = await acquire_hold(bus, holder="durable-runs:run-1", work_class="agent", request_id="run-1:1")
        ref = hold_ref(reply, "durable-runs:run-1")
        stale = GpuLeaseRefV1(lease_id=ref.lease_id, generation=ref.generation + 1, role=ref.role, holder=ref.holder)
        with pytest.raises(LeaseUnavailable) as err:
            async with gpu_lease(bus, work_class="agent", holder="orion-llm-gateway", hold=stale):
                pass
        assert err.value.reason.startswith("stale_hold_generation")
        assert (await rt.store.lease(ref.lease_id))["status"] == "granted"
        await validate_hold_ref(bus, ref, source="hub")          # the run still holds it
    asyncio.run(go())
