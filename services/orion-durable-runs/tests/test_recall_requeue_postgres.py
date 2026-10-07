"""A recall that runs out its grace re-queues the run's hold; it never fails the run (2026-09-29).

Live 2026-09-26..28 (``durable_resource_events``, ``event='run.failed'``): 19 of 39 failed durable
runs died on the pool taking a seat back, not on anything the run did:

* ``HoldLost: gpu_hold_lost:queued:recall_grace_exceeded`` (10) -- recalled (gpu2's max_hold seat
  limit, or chat's owner reclaiming lent gpu0), aborted after the 600 s grace, re-queued by the pool.
  Curiosity spent one of three attempts per recall; self-sense failed on the first.
* ``HoldLost: gpu_hold_lost:queued`` (7) -- the same thing answered by the pre-#2385 pool, whose
  ``queued`` reply carried no reason.
* ``HoldLost: gpu_hold_lost:unavailable:recall_grace_exceeded`` (2) -- the pool spent one of ITS three
  attempts per abort too, so the third recall dead-lettered the hold.

Spec (stage 4.3/4.5): "then it is aborted and re-queued under the same lease id"; "a lost hold stops
the turn and the run waits for the same lease id". Each case here fails on the pre-fix code with the
exact live error string.

Real Postgres (ORION_ADMISSION_TEST_DSN), real in-process pool, real client + codec.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orion.schemas.durable_run import DurableRunRequestV1  # noqa: E402
from orion.schemas.gpu_pool import GpuLeaseReplyV1, GpuLeaseRequestV1, GpuPoolControlV1  # noqa: E402
from orion.schemas.self_sense import SELF_SENSE_QUESTIONS  # noqa: E402
from pool_fixture import CFG, InProcessPool  # noqa: E402
from test_admission_runtime_postgres import (  # noqa: E402
    DSN, occupy_agent, request, runtime, until, with_database,
)
from test_pool_refusal_postgres import scripted  # noqa: E402

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
GRACE = CFG.defaults.hold_clawback_grace_sec


def self_sense_request(run_id: str) -> DurableRunRequestV1:
    return DurableRunRequestV1(run_id=run_id, workflow="self_sense_eval", correlation_id=run_id + "-corr",
                               admission={}, brief={
                                   "prompt": "self-sense eval: four fixed questions", "session_id": "self-sense-eval",
                                   "timeout_sec": 100, "source_tag": "curiosity_self_sense_eval",
                                   "line": "self_sense_eval", "questions": [list(q) for q in SELF_SENSE_QUESTIONS],
                                   "self_definition_version": 3, "lived_answers": []})


async def borrowed_chat_hold(gpu, rt, req):
    """The run's hold lands on lent gpu0 (chat) -- a borrowed card its owner can reclaim -- and the
    run's turn is in flight under it. Returns (driver task, hold row)."""
    await rt.submit(req)
    driver = asyncio.create_task(rt._drive(await rt.store.get_run(req.run_id)))
    await until(lambda: rt.runner.calls, attempts=1500)
    [hold] = [h for h in gpu.leases(holder=f"durable-runs:{req.run_id}") if h["status"] == "granted"]
    assert hold["role"] == "chat"
    return driver, hold


async def recall_past_grace(gpu, hold_id, blocker):
    """Juniper's chat wants gpu0 back and keeps it past the hold's 600 s grace: the pool aborts the
    hold. Returns the owner's lease id (still granted)."""
    owner = await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", work_class="chat", holder="hub-chat",
                                                 priority="interactive", request_id=f"owner-{hold_id}-{gpu.clock()}"))
    assert owner.status == "granted"
    await until(lambda: gpu.rt.store.leases[hold_id]["status"] == "recalling", attempts=1500)
    # every=20 < the owner's 30 s request TTL: later() advances the clock BEFORE it beats, and the
    # durable driver can run (and make the pool tick) in between -- a 30 s step would expire the owner.
    await gpu.later(GRACE - 30, beat=[blocker, owner.lease_id, hold_id], every=20)
    await gpu.later(60, beat=[blocker, owner.lease_id], every=20)
    assert gpu.rt.store.leases[hold_id]["status"] != "recalling"      # aborted past its grace
    return owner.lease_id


async def drive_to_terminal(rt, store, run_id, tries=5):
    """What the reconcile tick does: drive the run until it reaches a terminal state."""
    for _ in range(tries):
        row = await store.get_run(run_id)
        if row["terminal"]:
            return
        await rt.on_pool_event({"holder": f"durable-runs:{run_id}", "event": "granted"})
        await rt._drive(row)


async def failed_error(store, run_id):
    return [e["detail"].get("error") for e in await store.history(run_id) if e["event"] == "run.failed"]


def test_curiosity_run_recalled_past_grace_three_times_keeps_its_hold_and_completes():
    """Live ``HoldLost: gpu_hold_lost:unavailable:recall_grace_exceeded`` (and the two ``queued``
    failures before it): three recalls = three spent attempts on both sides = a failed run."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        blocker = await occupy_agent(gpu)
        await gpu.rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        req = request("recall-3x")
        driver, hold = await borrowed_chat_hold(gpu, rt, req)
        for cycle in range(1, 4):
            owner = await recall_past_grace(gpu, hold["lease_id"], blocker)
            await asyncio.wait_for(driver, 15)
            assert rt.runner.cancels(), "the turn under an aborted hold must be stopped"
            assert await failed_error(store, req.run_id) == [], cycle
            snap = await rt.graph.aget_state(rt.config(req.run_id))
            assert snap.values["hold"]["lease_id"] == hold["lease_id"], cycle      # same hold, same place
            assert snap.values["attempt"] == 0 and snap.values["status"] == "waiting_resource", cycle
            row = await gpu.lease(hold["lease_id"])
            assert row["status"] == "queued" and row["attempt"] == 1, (cycle, row)
            # Juniper's chat finishes: the hold is re-granted (next generation) and the node replays.
            await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=owner, outcome="ok"))
            await gpu.later(30, beat=[blocker])
            assert (await gpu.lease(hold["lease_id"]))["generation"] == cycle + 1
            await rt.on_pool_event({"holder": "durable-runs:recall-3x", "event": "granted"})
            calls = len(rt.runner.calls)
            driver = asyncio.create_task(rt._drive(await store.get_run(req.run_id)))
            await until(lambda: len(rt.runner.calls) > calls, attempts=1500)
        block.set()
        await asyncio.wait_for(driver, 15)
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [c.gpu_lease.generation for c in rt.runner.calls] == [1, 2, 3, 4]
        assert {c.gpu_lease.lease_id for c in rt.runner.calls} == {hold["lease_id"]}
        assert len(gpu.leases(holder="durable-runs:recall-3x")) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


def _strip_reason(gpu, lease_id):
    """The pre-#2385 pool: its ``queued`` reply carried no reason."""
    async def answer(req):
        reply = await gpu.dispatch(req)
        if req.lease_id == lease_id and reply.status == "queued":
            reply = reply.model_copy(update={"reason": None})
        return reply
    return answer


def _dead_lettered(gpu, lease_id):
    """The pre-fix pool after a third abort: the hold dead-lettered, answered as unavailable."""
    async def answer(req):
        reply = await gpu.dispatch(req)
        if req.lease_id == lease_id and req.verb in ("heartbeat", "status") and reply.status == "queued":
            reply = GpuLeaseReplyV1(status="unavailable", lease_id=lease_id, reason="recall_grace_exceeded")
        return reply
    return answer


@pytest.mark.parametrize("pool_answer,live_error", [
    ("current", "HoldLost: gpu_hold_lost:queued:recall_grace_exceeded"),
    ("pre_2385", "HoldLost: gpu_hold_lost:queued"),
    ("dead_lettered", "HoldLost: gpu_hold_lost:unavailable:recall_grace_exceeded"),
])
def test_self_sense_run_recalled_past_grace_waits_for_a_hold_instead_of_failing(pool_answer, live_error):
    """Self-sense failed on the FIRST lost hold. Each pool answer live runs saw after the abort must
    leave the run waiting (the same hold when the pool kept it; a fresh one when it ended it), then
    finish -- never ``run.failed`` with the live error string."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        blocker = await occupy_agent(gpu)
        await gpu.rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        run_id = f"ss-recall-{pool_answer}"
        req = self_sense_request(run_id)
        driver, hold = await borrowed_chat_hold(gpu, rt, req)
        if pool_answer != "current":
            bus = scripted(rt)
            make = _strip_reason if pool_answer == "pre_2385" else _dead_lettered
            for verb in ("heartbeat", "status"):
                bus.script[verb] = make(gpu, hold["lease_id"])
        owner = await recall_past_grace(gpu, hold["lease_id"], blocker)
        await asyncio.wait_for(driver, 15)
        assert await failed_error(store, req.run_id) == [], live_error
        assert (await store.get_run(req.run_id))["terminal"] is None
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert snap.next == ("resource_wait",) and snap.values["status"] == "waiting_resource"
        if pool_answer == "dead_lettered":
            # The pool ended that hold: the run asks afresh under a new request id.
            assert snap.values["hold"]["lease_id"] != hold["lease_id"]
            assert snap.values["hold"]["request_id"].endswith(":2")
        else:
            assert snap.values["hold"]["lease_id"] == hold["lease_id"]
        await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=owner, outcome="ok"))
        await gpu.later(30, beat=[blocker])
        await rt.on_pool_event({"holder": f"durable-runs:{run_id}", "event": "granted"})
        block.set()
        await drive_to_terminal(rt, store, req.run_id)
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_step_that_never_fits_fails_at_the_take_back_limit_instead_of_replaying_forever():
    """DURABLE_RUNS_HOLD_MAX_TAKEBACKS bounds the replay loop (take-backs are not attempts)."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        blocker = await occupy_agent(gpu)
        await gpu.rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
        rt = runtime(pool, saver, store, asyncio.Event(), gpu=gpu, DURABLE_RUNS_HOLD_MAX_TAKEBACKS=1)
        req = request("recall-limit")
        driver, hold = await borrowed_chat_hold(gpu, rt, req)
        owner = await recall_past_grace(gpu, hold["lease_id"], blocker)          # take-back 1: waits
        await asyncio.wait_for(driver, 15)
        assert await failed_error(store, req.run_id) == []
        await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=owner, outcome="ok"))
        await gpu.later(30, beat=[blocker])
        await rt.on_pool_event({"holder": "durable-runs:recall-limit", "event": "granted"})
        driver = asyncio.create_task(rt._drive(await store.get_run(req.run_id)))
        await until(lambda: len(rt.runner.calls) == 2, attempts=1500)
        await recall_past_grace(gpu, hold["lease_id"], blocker)                   # take-back 2: over
        await asyncio.wait_for(driver, 15)
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        [error] = await failed_error(store, req.run_id)
        assert error.startswith("hold_takeback_limit:1: HoldLost: gpu_hold_lost:queued:recall_grace_exceeded")
        assert (await gpu.lease(hold["lease_id"]))["status"] == "released"      # the seat is given back
        await rt.close()
    asyncio.run(with_database(scenario))
