"""Stage 5.4 end to end: world-model and the visual chain share gpu2 only through the pool.

The real client (orion.gpu_pool.client) against a real in-process pool and the real
config/gpu_pool.yaml (`world.serialize_with: [diffusion]`), over test_client_roundtrip's WiredBus
(real codec both ways). Runs on the in-memory store and, when GPU_POOL_TEST_POSTGRES_URI is set, on
the real Postgres projection + LangGraph saver (CI provides one).

What it pins -- the guarantee the durable-runs /capacity permit gave, now kept by the pool:
- a world lease is never granted while a diffusion hold (a reverie-visual run) is held; it says
  why (queued reason=serialized:diffusion) and gives up at its short deadline (world-model maps
  that to gpu_contended);
- a world lease that can wait gets the card once the hold is released (class world waits);
- a diffusion hold queued behind a live world lease waits (serialized:world) and is granted when
  the world lease ends;
- at no observed point are a world lease and a diffusion lease both granted.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from orion.gpu_pool.client import (
    LeaseUnavailable, acquire_hold, durable_run_holder, gpu_lease, hold_ref, lease_status, release_lease,
)
from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL

from tests.test_client_roundtrip import WiredBus, pool

PG_URI = os.environ.get("GPU_POOL_TEST_POSTGRES_URI")
BACKENDS = ["memory", pytest.param("postgres", marks=pytest.mark.skipif(
    not PG_URI, reason="GPU_POOL_TEST_POSTGRES_URI not set"))]
HOLDER = durable_run_holder("reverie-visual-e2e")


async def _runtime(bus, backend):
    if backend == "memory":
        rt, _ = await pool(bus)
        return rt, None
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from app.store import PostgresStore
    from tests.test_store_postgres import _pool

    pg = await _pool()
    saver = AsyncPostgresSaver(pg)
    await saver.setup()
    rt, _ = await pool(bus, store=PostgresStore(pg), saver=saver)
    return rt, pg


async def _granted_roles(rt) -> list[str]:
    return sorted(r["role"] for r in await rt.store.live_leases()
                  if r["status"] in ("granted", "recalling") and r.get("role") in ("world", "diffusion"))


def _assert_never_both(roles: list[str]) -> None:
    assert not ("world" in roles and "diffusion" in roles), f"world and diffusion granted together: {roles}"


async def _drain(events: asyncio.Queue, bus) -> list[dict]:
    out = []
    while not events.empty():
        out.append(bus.codec.decode((await events.get())["data"]).envelope.payload)
    return out


@pytest.mark.parametrize("backend", BACKENDS)
def test_world_and_diffusion_never_compute_together_on_gpu2(backend):
    async def go():
        bus = WiredBus()
        rt, pg = await _runtime(bus, backend)
        try:
            async with bus.subscribe(GPU_POOL_EVENT_CHANNEL) as events:
                # 1. A reverie-visual run holds diffusion on gpu2.
                hold = await acquire_hold(bus, holder=HOLDER, work_class="diffusion", request_id="rv:1")
                assert hold.status == "granted" and hold.grant.role == "diffusion"
                _assert_never_both(await _granted_roles(rt))

                # 2. A world prediction with world-model's short deadline gives up, and says why.
                with pytest.raises(LeaseUnavailable) as refused:
                    async with gpu_lease(bus, work_class="world", holder="world-model", priority="system",
                                         deadline_sec=0.3):
                        raise AssertionError("world must not be granted while diffusion is held")
                assert refused.value.reason == "deadline"          # world-model -> gpu_contended
                seen = await _drain(events, bus)
                serialized = [e for e in seen if e.get("event") == "queued" and e.get("work_class") == "world"
                              and e.get("reason") == "serialized:diffusion"]
                assert serialized and serialized[0]["detail"] == {"serialized": True}
                world_row = await rt.store.lease(refused.value.lease_id)
                assert world_row["status"] not in ("queued", "granted"), "a late world lease must not linger"
                _assert_never_both(await _granted_roles(rt))

                # 3. A world lease that can wait gets gpu2 as soon as the hold is released.
                async def world_waits():
                    async with gpu_lease(bus, work_class="world", holder="world-model", deadline_sec=5) as lease:
                        roles = await _granted_roles(rt)
                        _assert_never_both(roles)
                        return lease.grant.role

                waiter = asyncio.create_task(world_waits())
                await asyncio.sleep(0.1)
                assert not waiter.done()
                await release_lease(bus, hold.lease_id, source=HOLDER)
                assert await asyncio.wait_for(waiter, 5) == "world"

                # 4. The other direction: a diffusion hold queued behind a live world lease waits.
                world_done = asyncio.Event()

                async def world_holds():
                    async with gpu_lease(bus, work_class="world", holder="world-model", deadline_sec=5):
                        await world_done.wait()

                holder = asyncio.create_task(world_holds())
                await asyncio.sleep(0.1)
                await _drain(events, bus)
                second = await acquire_hold(bus, holder=HOLDER, work_class="diffusion", request_id="rv:2")
                assert second.status == "queued"
                seen = await _drain(events, bus)
                assert any(e.get("event") == "queued" and e.get("lease_id") == second.lease_id
                           and e.get("reason") == "serialized:world" for e in seen)
                _assert_never_both(await _granted_roles(rt))
                world_done.set()
                await asyncio.wait_for(holder, 5)
                await rt.tick()
                status = await lease_status(bus, second.lease_id, source=HOLDER)
                assert status.status == "granted" and status.grant.role == "diffusion"
                _assert_never_both(await _granted_roles(rt))
                await release_lease(bus, second.lease_id, source=HOLDER)
        finally:
            if pg is not None:
                await pg.close()
    asyncio.run(go())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_render_outlives_its_released_hold_and_still_keeps_world_off(backend):
    """Review finding (stage 5.4): the durable run gives its hold back on a step deadline or recall
    while the diffusion thread is still rendering. The generate's attached child lease must keep
    world off gpu2 until the render really ends -- releasing the hold alone must not free the card."""
    async def go():
        bus = WiredBus()
        rt, pg = await _runtime(bus, backend)
        try:
            hold = await acquire_hold(bus, holder=HOLDER, work_class="diffusion", request_id="rv:9")
            ref = hold_ref(hold, HOLDER)
            render_done = asyncio.Event()

            async def render():   # what orion-thought's generate_visual_bytes does under a hold
                async with gpu_lease(bus, work_class="diffusion", holder="orion-thought", hold=ref,
                                     deadline_sec=5) as child:
                    assert child.grant.role == "diffusion"
                    await render_done.wait()

            renderer = asyncio.create_task(render())
            await asyncio.sleep(0.1)
            await release_lease(bus, hold.lease_id, source=HOLDER, outcome="cancelled")   # run gives up (step deadline)
            await rt.tick()

            with pytest.raises(LeaseUnavailable):
                async with gpu_lease(bus, work_class="world", holder="world-model", deadline_sec=0.3):
                    raise AssertionError("world granted while the render (child lease) still runs")

            render_done.set()
            await asyncio.wait_for(renderer, 5)
            async with gpu_lease(bus, work_class="world", holder="world-model", deadline_sec=2) as lease:
                assert lease.grant.role == "world"
        finally:
            if pg is not None:
                await pg.close()
    asyncio.run(go())
