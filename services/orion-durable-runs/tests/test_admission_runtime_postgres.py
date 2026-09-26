"""Isolated real Postgres checkpoints and run registry, driven on the REAL GPU pool (stage 4.5).

Set ORION_ADMISSION_TEST_DSN to a disposable database. Each test uses a fresh schema; this test
never connects to POSTGRES_URI or production configuration. The pool is the real pool runtime
in process (tests/pool_fixture.py) with a fake clock and fixture llama.cpp servers; the lease
RPCs go through the real client and codec.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from app.admission_runtime import AdmissionRuntime
from app.graph import Deps
from app.pool_hold import PoolHolds
from app.settings import Settings
from orion.durable_admission.store import PostgresAdmissionStore
from orion.schemas.durable_run import DurableRunRequestV1, CuriosityTurnResultV1
from orion.schemas.gpu_pool import GpuLeaseRequestV1, GpuPoolControlV1
from pool_fixture import CFG, InProcessPool, PoolBus

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


class Runner:
    def __init__(self, saver, block=None, outcome=None):
        self._checkpointer = saver
        self.calls = []
        self.events = []
        self.block = block
        self.outcome = outcome
        self.reflect_calls = []
        self._bus = None

    def _curiosity_deps(self):
        async def turn(req):
            self.calls.append(req)
            if self.block:
                await self.block.wait()
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="Evidence-backed finding")
        async def read(run_id):
            return {"graph_readable": True, **({"outcome": self.outcome} if self.outcome else {})}
        async def row(facts):
            return True
        async def journal(entry):
            return entry.entry_id
        return Deps(turn, read, row, journal)

    def _self_sense_deps(self):
        from app.self_sense_graph import Deps as SelfSenseDeps

        async def turn(req):
            self.calls.append(req)
            if self.block:
                await self.block.wait()
            return CuriosityTurnResultV1(
                run_id=req.run_id, correlation_id=req.correlation_id, text="self-sense answer", ok=True
            )

        async def publish_rows(rows):
            return len(rows), 0

        return SelfSenseDeps(run_turn=turn, publish_rows=publish_rows)

    def _reflect_deps(self):
        from app.reflect_graph import Deps as ReflectDeps

        async def call(reflect_input, llm_route, gpu_lease=None):
            self.reflect_calls.append((llm_route, gpu_lease))
            return [{"kind": "fixture", "summary": "reflect fixture"}]

        return ReflectDeps(call_reflect_llm=call)

    async def _publish(self, channel, kind, model, corr):
        self.events.append((kind, model))
        return True

    async def _emit_state(self, *args, **kwargs):
        self.events.append(("state", kwargs))

    def _corr_for_admission(self, value):
        return value

    def cancels(self):
        return [model.correlation_id for kind, model in self.events if kind == "harness.run.cancel.v1"]


def request(run_id, *, workflow="curiosity.investigate", timeout=100, **admission):
    brief = {"prompt": "Inspect the current self-inquiry evidence.", "session_id": "curiosity", "timeout_sec": timeout}
    if workflow == "self_study.reflect":
        brief.update(line="reflect", llm_route="agent", self_study_reflect_input={"snapshot_id": "s1"})
    return DurableRunRequestV1(run_id=run_id, workflow=workflow, correlation_id=str(uuid4()),
        admission=admission, brief=brief)


async def with_database(scenario):
    schema = "admission_test_"+uuid4().hex
    async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    async with AsyncConnectionPool(DSN, min_size=1, max_size=10, open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema},public"}) as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        store = PostgresAdmissionStore(pool)
        await store.setup()
        await scenario(pool, saver, store)


def runtime(pool, saver, store, block=None, *, gpu=None, outcome=None, **overrides):
    """An AdmissionRuntime on the real in-process pool ``gpu`` (booted by the caller)."""
    settings = Settings(**{"_env_file": None, "POSTGRES_URI": DSN, "ORION_BUS_ENABLED": False,
                           "DURABLE_RUNS_ADMISSION_ENABLED": True, "DURABLE_RUNS_LEASE_HEARTBEAT_SEC": 0.03,
                           "DURABLE_RUNS_RETRY_BASE_SEC": 0.05, "DURABLE_RUNS_HOLD_STATUS_POLL_SEC": 60, **overrides})
    runner = Runner(saver, block, outcome)
    gpu = gpu or InProcessPool()
    rt = AdmissionRuntime(settings, runner, pool, store=store,
                          holds=PoolHolds(PoolBus(gpu), source="orion-durable-runs", cfg=CFG))
    rt.gpu = gpu
    return rt


async def legacy_rows(store):
    async with store.pool.connection() as conn:
        demands = (await (await conn.execute("SELECT count(*) AS n FROM durable_resource_demands")).fetchone())["n"]
        leases = (await (await conn.execute("SELECT count(*) AS n FROM durable_resource_leases")).fetchone())["n"]
    return demands, leases


async def occupy_agent(gpu, name="blocker"):
    """Another durable run holding the agent card (the only card class agent can use today)."""
    reply = await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", request_id=f"{name}:1", kind="hold",
        holder=f"durable-runs:{name}", work_class="agent", priority="background", retryable=True))
    assert reply.status == "granted" and reply.grant.role == "agent"
    return reply.lease_id


def pool_event(gpu, lease_id, name):
    return next(e for e in gpu.events(name) if e.get("lease_id") == lease_id)


async def until(predicate, *, attempts=300):
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def test_waiting_run_queues_in_the_pool_wakes_on_the_grant_and_completes_under_its_hold():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        blocker = await occupy_agent(gpu)
        rt = runtime(pool, saver, store, gpu=gpu)
        waiting = request("waiting-001", requirements={"minimum_context_tokens": 32768})
        assert (await asyncio.wait_for(rt.submit(waiting), 1))["status"] == "waiting_resource"
        await rt._drive(await store.get_run(waiting.run_id))
        assert not rt.active and not rt.runner.calls
        snap = await rt.graph.aget_state(rt.config(waiting.run_id))
        assert snap.next == ("resource_wait",)
        [hold] = gpu.leases(holder="durable-runs:waiting-001")
        assert hold["kind"] == "hold" and hold["status"] == "queued" and hold["work_class"] == "agent"
        assert hold["priority"] == "background" and hold["request_id"] == "waiting-001:1"
        assert snap.values["hold"] == {"request_id": "waiting-001:1", "lease_id": hold["lease_id"]}

        # No polling storm: a waiting run is not re-read before its poll interval without an event.
        before = gpu.verbs().count("status")
        for _ in range(5):
            await rt._drive(await store.get_run(waiting.run_id))
        assert gpu.verbs().count("status") == before
        await rt.close()

        # Restart; the blocker finishes; the pool grants and publishes it; the event wakes the run.
        restarted = runtime(pool, saver, store, gpu=gpu)
        await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=blocker, outcome="ok"))
        granted = pool_event(gpu, hold["lease_id"], "granted")
        await restarted.on_pool_event(granted)
        await restarted._drive(await store.get_run(waiting.run_id))
        assert len(restarted.runner.calls) == 1
        turn = restarted.runner.calls[0]
        assert turn.gpu_lease.lease_id == hold["lease_id"] and turn.gpu_lease.holder == "durable-runs:waiting-001"
        assert turn.assigned_lane is None and turn.lease is None
        assert (await store.get_run(waiting.run_id))["terminal"] == "completed"
        assert (await gpu.lease(hold["lease_id"]))["status"] == "released"
        assert (await restarted.submit(waiting))["status"] == "completed"
        await restarted.reconcile()  # durable event outbox can be replayed
        await restarted.close()
        events = {e["event"]: e for e in await store.history(waiting.run_id)}
        assert {"run.waiting_resource", "run.resource_granted", "run.lane_assigned", "run.started",
                "resource.lease_released", "run.completed"} <= set(events)
        assert events["run.lane_assigned"]["detail"]["lane"] == "agent"
        # Acceptance check 1: the durable broker tables get nothing new; the pool holds one hold.
        assert await legacy_rows(store) == (0, 0)
        assert len(gpu.leases(holder="durable-runs:waiting-001", kind="hold")) == 1
    asyncio.run(with_database(scenario))


def test_missed_event_falls_back_to_one_status_read_per_poll_interval():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        blocker = await occupy_agent(gpu)
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.05)
        req = request("missed-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=blocker, outcome="ok"))
        await asyncio.sleep(0.06)   # the granted event is never delivered
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_cancel_running_stops_the_turn_and_releases_the_hold():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        req = request("cancel-001")
        await rt.submit(req)
        await rt.reconcile()
        await until(lambda: rt.runner.calls)
        assert (await rt.status(req.run_id))["status"] == "running"
        [hold] = gpu.leases(holder="durable-runs:cancel-001")
        assert hold["status"] == "granted"
        await rt.control(req.run_id, "cancel")
        assert (await rt.control(req.run_id, "resume"))["status"] == "cancelled"
        assert (await gpu.lease(hold["lease_id"]))["status"] == "released"
        assert rt.runner.cancels(), "the running harness turn must be cancelled"
        await rt._drive(await store.get_run(req.run_id))
        assert (await rt.status(req.run_id))["status"] == "cancelled"
        assert not (await rt.graph.aget_state(rt.config(req.run_id))).next
        await rt.close()
    asyncio.run(with_database(scenario))


def test_lost_heartbeat_mid_turn_cancels_the_turn_and_waits_for_the_same_lease():
    """Acceptance check 6 (durable side): the hold expires while the turn runs -> the turn is
    stopped, the run keeps its lease_id and place, and it finishes when the pool re-grants it
    (generation + 1)."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        req = request("expire-001")
        await rt.submit(req)
        driver = asyncio.create_task(rt._drive(await store.get_run(req.run_id)))
        await until(lambda: rt.runner.calls)
        [hold] = gpu.leases(holder="durable-runs:expire-001")
        # The pool's clock passes the hold TTL with no heartbeat landing (fake clock jump).
        await gpu.later(CFG.defaults.hold_lease_ttl_sec + 1)
        assert (await gpu.lease(hold["lease_id"]))["status"] == "retry_wait"
        await asyncio.wait_for(driver, 5)
        assert rt.runner.cancels(), "a turn under a lost hold must be stopped"
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert snap.values["hold"]["lease_id"] == hold["lease_id"]       # kept: same lease, same place
        assert snap.values["lease"] is None and snap.values["status"] == "retrying"
        history = [e["event"] for e in await store.history(req.run_id)]
        assert "resource.lease_expired" in history and "resource.lease_released" not in history
        block.set()
        await gpu.later(60)                                               # retry delay: re-queued, re-granted
        row = await gpu.lease(hold["lease_id"])
        assert row["status"] == "granted" and row["generation"] == 2
        await asyncio.sleep(0.06)                                         # durable retry backoff
        await rt.on_pool_event({"holder": "durable-runs:expire-001", "event": "granted"})
        await rt._drive(await store.get_run(req.run_id))
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [c.gpu_lease.generation for c in rt.runner.calls] == [1, 2]
        assert {c.gpu_lease.lease_id for c in rt.runner.calls} == {hold["lease_id"]}
        assert len(gpu.leases(holder="durable-runs:expire-001")) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


def test_restart_mid_turn_fences_the_old_turn_and_replays_under_the_same_hold():
    """Acceptance check 6: durable-runs dies mid-hold and restarts inside the TTL. The new driver
    cancels the old turn's identity, then replays under the SAME lease_id and generation with a new
    turn identity (turn_fence) -- a delayed cancel can never hit the replay."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        block = asyncio.Event()
        dead = runtime(pool, saver, store, block, gpu=gpu)
        req = request("restart-001")
        await dead.submit(req)
        driver = asyncio.create_task(dead._drive(await store.get_run(req.run_id)))
        await until(lambda: dead.runner.calls)
        old_turn = dead.runner.calls[0].correlation_id
        # A crash: no graceful harness cancel and no release reach anyone.
        async def crashed(*args, **kwargs):
            return None
        dead._cancel_harness = crashed
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        [hold] = gpu.leases(holder="durable-runs:restart-001")
        assert hold["status"] == "granted"

        block.set()
        alive = runtime(pool, saver, store, gpu=gpu)
        await alive._drive(await store.get_run(req.run_id))
        assert old_turn in alive.runner.cancels()
        [replay] = alive.runner.calls
        assert replay.correlation_id != old_turn
        assert (replay.gpu_lease.lease_id, replay.gpu_lease.generation) == (hold["lease_id"], 1)
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert len(gpu.leases(holder="durable-runs:restart-001")) == 1
        await alive.close()
    asyncio.run(with_database(scenario))


def test_recall_inside_grace_releases_at_the_next_node_boundary():
    """A hold borrowing lent gpu0 is recalled when chat wants its card back: the run finishes its
    current node, then lets go at the boundary -- well inside hold_clawback_grace_sec, never aborted."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        await occupy_agent(gpu)
        await gpu.rt.control(GpuPoolControlV1(verb="lend", card="gpu0"))
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block, gpu=gpu)
        req = request("recall-001")
        await rt.submit(req)
        driver = asyncio.create_task(rt._drive(await store.get_run(req.run_id)))
        await until(lambda: rt.runner.calls)
        [hold] = gpu.leases(holder="durable-runs:recall-001")
        assert hold["role"] == "chat" and rt.runner.calls[0].gpu_lease.role == "chat"
        owner = await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", work_class="chat", holder="hub-chat",
                                                     priority="interactive", request_id="owner-1"))
        assert owner.status == "granted"                  # Juniper's chat is not kept waiting
        await until(lambda: gpu.rt.store.leases[hold["lease_id"]]["status"] == "recalling")
        block.set()
        await asyncio.wait_for(driver, 5)
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        row = await gpu.lease(hold["lease_id"])
        assert row["status"] == "released"
        assert not [e for e in gpu.events("aborted") if e.get("lease_id") == hold["lease_id"]]
        released = [e for e in await store.history(req.run_id) if e["event"] == "resource.lease_released"]
        assert released[0]["detail"]["reason"] == "recalled"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_door_a_keeps_heartbeating_the_hold_until_hub_releases_it_and_a_restart_adopts_it():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, outcome={"reach_out": True, "reach_out_why": "she should know"})
        req = request("door-a-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        [hold] = gpu.leases(holder="durable-runs:door-a-001")
        assert hold["status"] == "granted" and req.run_id in rt.outreach
        completed = next(e for e in await store.history(req.run_id) if e["event"] == "run.completed")
        assert completed["detail"]["gpu_lease"] == {"lease_id": hold["lease_id"], "generation": 1,
                                                    "role": "agent", "holder": "durable-runs:door-a-001"}
        assert "resource_lease" not in completed["detail"]
        beats = gpu.verbs().count("heartbeat")
        await asyncio.sleep(0.05)
        await rt.reconcile()
        assert gpu.verbs().count("heartbeat") > beats
        # A restarted process adopts the Door-A hold from the outbox history.
        await rt.close()
        restarted = runtime(pool, saver, store, gpu=gpu)
        await restarted.reconcile()
        assert req.run_id in restarted.outreach
        assert (await restarted.release_outreach(req.run_id))["released"] is True
        assert (await gpu.lease(hold["lease_id"]))["status"] == "released"
        assert (await restarted.release_outreach(req.run_id))["reason"] == "already_released"
        await restarted.close()
    asyncio.run(with_database(scenario))


def test_admitted_reflect_run_holds_a_gpu_and_its_llm_call_carries_the_hold():
    """Stage 4.4 hazard: the reflect call used to carry neither lease (it would queue behind its
    own run once runs hold). It now runs under the run's hold and passes the ref."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("reflect-001", workflow="self_study.reflect")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        [(route, ref)] = rt.runner.reflect_calls
        [hold] = gpu.leases(holder="durable-runs:reflect-001")
        assert route == "agent" and ref.lease_id == hold["lease_id"] and ref.role == "agent"
        assert hold["status"] == "released"
        completed = next(e for e in await store.history(req.run_id) if e["event"] == "run.completed")
        assert completed["detail"]["line"] == "reflect" and completed["detail"]["llm_call_ok"] is True
        assert not rt.runner.calls, "a reflect run must never drive a curiosity harness turn"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_legacy_durable_lease_in_a_checkpoint_is_dropped_and_the_run_asks_the_pool():
    """A checkpoint written before the cutover can still name a durable-runs lease. It is never sent
    to Hub again; the run takes a pool hold instead."""
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.01)
        req = request("legacy-001")
        await rt.submit(req)
        blocker = await occupy_agent(gpu)
        await rt._drive(await store.get_run(req.run_id))   # checkpointed, waiting behind the blocker
        [first] = gpu.leases(holder="durable-runs:legacy-001")
        await gpu.dispatch(GpuLeaseRequestV1(verb="cancel", lease_id=first["lease_id"]))
        await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=blocker, outcome="ok"))
        cfg = rt.config(req.run_id)
        legacy = {"run_id": req.run_id, "demand_id": f"{req.run_id}:harness_turn:llm.route.agent", "lease_id": "old",
                  "resource_key": "llm.route.agent", "lane": "agent-burst", "backend_key": "http://100.112.254.99:8016",
                  "generation": 7, "status": "active"}
        await rt.graph.aupdate_state(cfg, {"lease": legacy, "hold": None, "status": "admitted"}, as_node="run_started")
        assert (await rt.graph.aget_state(cfg)).next == ("harness_turn",)
        await rt._drive(await store.get_run(req.run_id))   # recovery -> retry_wait -> a new pool hold
        await asyncio.sleep(0.02)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        [turn] = rt.runner.calls
        assert turn.lease is None and turn.gpu_lease is not None and turn.assigned_lane is None
        assert "agent-burst" not in turn.model_dump_json()
        assert await legacy_rows(store) == (0, 0)
        await rt.close()
    asyncio.run(with_database(scenario))


def test_unknown_route_fails_the_run_with_the_reason_instead_of_waiting_forever():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        req = request("route-001", preferred_lane="no-such-route", resource="llm.route.no-such-route")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        row = await store.get_run(req.run_id)
        assert row["terminal"] == "failed"
        failed = next(e for e in await store.history(req.run_id) if e["event"] == "run.failed")
        assert failed["detail"]["error"] == "gpu_pool_unknown_route:no-such-route"
        assert not gpu.leases()
        await rt.close()
    asyncio.run(with_database(scenario))


def test_unreachable_pool_keeps_the_request_id_and_retries_idempotently():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.01)
        rt.holds.bus.fail_next = 1
        req = request("unreachable-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert snap.next == ("resource_wait",)
        assert snap.values["hold"] == {"request_id": "unreachable-001:1", "lease_id": None}
        assert (await store.get_run(req.run_id))["terminal"] is None
        await asyncio.sleep(0.02)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [r["request_id"] for r in gpu.leases(holder="durable-runs:unreachable-001")] == ["unreachable-001:1"]
        await rt.close()
    asyncio.run(with_database(scenario))


def test_heartbeat_interval_must_fit_twice_in_the_pool_hold_ttl():
    async def scenario(pool, saver, store):
        with pytest.raises(ValueError, match="at most half"):
            runtime(pool, saver, store, DURABLE_RUNS_LEASE_HEARTBEAT_SEC=46, DURABLE_RUNS_LEASE_SECONDS=120)
    asyncio.run(with_database(scenario))
