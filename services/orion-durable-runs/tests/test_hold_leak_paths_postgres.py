"""No GPU pool hold is left behind for a run that no longer needs it (4.5 review findings).

A leaked retryable hold is worse than it looks: the pool expires it into retry_wait, re-queues it
and grants it to nobody again, up to its retry budget. Real Postgres, real in-process pool.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from langgraph.graph import START

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.admitted_graph import HoldRecalled  # noqa: E402
from orion.schemas.gpu_pool import GpuLeaseRequestV1, GpuPoolControlV1  # noqa: E402
from pool_fixture import CFG, InProcessPool  # noqa: E402
from test_admission_runtime_postgres import DSN, occupy_agent, request, runtime, with_database  # noqa: E402

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def hold_req(run_id, seq=1):
    return GpuLeaseRequestV1(verb="acquire", request_id=f"{run_id}:{seq}", kind="hold", holder=f"durable-runs:{run_id}",
                             work_class="agent", priority="background", retryable=True)


def test_a_failed_release_rpc_is_retried_until_the_pool_confirms():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("leak-release-001")
        await rt.submit(req)
        held = await gpu.dispatch(hold_req(req.run_id))
        rt.holds.bus.fail_next = 1
        assert await rt._end_hold({"run_id": req.run_id}, held.lease_id, None, "completed") is False
        assert (await gpu.lease(held.lease_id))["status"] == "granted"      # still held at the pool
        await rt.reconcile()
        assert (await gpu.lease(held.lease_id))["status"] == "released" and not rt._pending_release
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_hold_dead_lettered_during_an_outage_is_replaced_not_fatal():
    """durable-runs down for minutes: the granted hold expires and is re-granted until the pool
    dead-letters it. On resume the run asks again under a new request id and completes."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("leak-deadletter-001")
        await rt.submit(req)
        blocker = await occupy_agent(gpu)
        await rt._drive(await store.get_run(req.run_id))               # queued behind the blocker
        [first] = gpu.leases(holder=f"durable-runs:{req.run_id}")
        await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=blocker, outcome="ok"))
        for _ in range(8):                                               # nobody heartbeats it
            await gpu.later(CFG.defaults.hold_lease_ttl_sec + 5)
        assert (await gpu.lease(first["lease_id"]))["status"] == "dead_letter"
        await rt.on_pool_event({"holder": f"durable-runs:{req.run_id}", "event": "dead_lettered"})
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [r["request_id"] for r in gpu.leases(holder=f"durable-runs:{req.run_id}")] == \
            [f"{req.run_id}:1", f"{req.run_id}:2"]
        await rt.close()
    asyncio.run(with_database(scenario))


def test_work_never_starts_on_a_hold_already_being_recalled():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        await occupy_agent(gpu)
        await gpu.rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("leak-recall-001")
        await rt.submit(req)
        held = await gpu.dispatch(hold_req(req.run_id))
        assert held.grant.role == "chat"
        await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", work_class="chat", holder="hub-chat",
                                             priority="interactive", request_id="owner-1"))
        await gpu.rt.heartbeat(held.lease_id)
        assert (await gpu.lease(held.lease_id))["status"] == "recalling"
        state = {"run_id": req.run_id, "correlation_id": req.correlation_id, "workflow": "curiosity.investigate",
                 "brief": req.brief.model_dump(mode="json"), "admission": req.admission.model_dump(mode="json"),
                 "lease": {"lease_id": held.lease_id, "generation": held.grant.generation, "role": "chat",
                           "holder": f"durable-runs:{req.run_id}"}}
        started = []

        async def node(_state):
            started.append(True)
            return {}

        with pytest.raises(HoldRecalled):
            await rt.execute(state, node)
        assert not started and (await gpu.lease(held.lease_id))["status"] == "released"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_cancel_during_an_in_flight_acquire_ends_the_hold_that_landed():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("leak-cancel-001")
        await rt.submit(req)
        cfg = rt.config(req.run_id)
        # The checkpoint exists, the acquire reached the pool, its reply never reached the state.
        await rt.graph.aupdate_state(cfg, {"run_id": req.run_id, "correlation_id": req.correlation_id,
            "brief": req.brief.model_dump(mode="json"), "admission": req.admission.model_dump(mode="json"),
            "requested_at": req.requested_at.isoformat(), "attempt": 0, "status": "queued",
            "workflow": "curiosity.investigate"}, as_node=START)
        landed = await gpu.dispatch(hold_req(req.run_id))
        await rt.control(req.run_id, "cancel")
        assert (await gpu.lease(landed.lease_id))["status"] == "released"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_door_a_hold_is_ended_when_the_run_does_not_end_completed():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("leak-door-a-001")
        await rt.submit(req)
        held = await gpu.dispatch(hold_req(req.run_id))
        ref = {"lease_id": held.lease_id, "generation": 1, "role": "agent", "holder": f"durable-runs:{req.run_id}"}
        await rt.keep_for_outreach({"run_id": req.run_id, "lease": ref})
        await store.set_control(req.run_id, "cancelled")                 # a cancel wins the terminal race
        await rt._terminal(req.run_id, "completed", {"run_id": req.run_id, "text": "x", "lease": ref})
        assert (await store.get_run(req.run_id))["terminal"] == "cancelled"
        assert req.run_id not in rt.outreach
        assert (await gpu.lease(held.lease_id))["status"] == "released"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_paused_run_is_not_re_driven_every_tick():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("leak-pause-001")
        await rt.submit(req)
        await occupy_agent(gpu)
        await rt._drive(await store.get_run(req.run_id))
        await rt.control(req.run_id, "pause")
        before = len(gpu.requests)
        for _ in range(3):
            await rt.reconcile()
        await asyncio.gather(*rt.active.values())
        assert len(gpu.requests) == before
        await rt.close()
    asyncio.run(with_database(scenario))
