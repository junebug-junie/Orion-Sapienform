"""Connected receipt-to-artifact acceptance on the GPU pool (stage 4.5): real service adapters,
real Postgres, and the REAL pool runtime in process.

cortex-orch dispatch -> durable-runs receipt -> the run asks the pool for a hold and waits (another
run holds the agent card) -> durable-runs restarts -> the pool grants the hold (the other run ends,
or -- ``gpu2`` -- the pool loads the second 27B on gpu2 after the run waited 1200 s, and an actuator
fixture answers its GpuActuateV1) -> the pool's ``granted`` event wakes the run over the bus -> the
turn runs through the production Hub, Thought, Governor, Cortex Exec and Gateway adapters, and the
gateway places every model call on the pool as an ``attach`` under the run's hold -> journal,
attention row and completion -> the hold is released.

Only model output, FCC's subprocess, the llama.cpp servers and the gpu2 actuator are fixtures.
"""
import asyncio
import importlib
import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI

from app.admission_runtime import AdmissionRuntime
from app.runner import DurableRunner
from app.settings import Settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.durable_run import (
    DurableRunRequestV1, CURIOSITY_TURN_REQUEST_CHANNEL, DURABLE_RUN_REQUEST_CHANNEL,
    DURABLE_RUN_STATE_KIND,
)
from orion.schemas.gpu_pool import (
    GPU_ACTUATE_RESULT_KIND, GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_POOL_ACTUATE_RESULT_CHANNEL, GPU_POOL_EVENT_CHANNEL, GpuActuateResultV1,
    GpuActuateV1, GpuLeaseRequestV1,
)
from orion.schemas.resource_admission import RESOURCE_EVENT_CHANNEL
from orion.schemas.attention_schema import ATTENTION_SCHEMA_KIND
from .acceptance_bus import TypedBus
from .acceptance_turn import build_turn_adapter, DRAFT, REPAIRED
from .pool_fixture import CFG, LIVE, InProcessPool
from .test_admission_runtime_postgres import DSN, legacy_rows, with_database

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
ROOT = Path(__file__).resolve().parents[3]
SEAT = "agent-gpu2"


def cortex_dispatch():
    spec = importlib.util.spec_from_file_location("acceptance_cortex_dispatch",
        ROOT / "services/orion-cortex-orch/app/durable_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dispatch_durable_run


def install_gpu2_actuator(bus, gpu, actions):
    """circe's controller, as a fixture: answers each GpuActuateV1 accepted -> succeeded, and the
    27B then answers discovery on gpu2 (diffusion stopped)."""
    async def actuator(env):
        msg = GpuActuateV1.model_validate(env.payload)
        actions.append(msg)

        async def answer(status, **kw):
            await bus.publish(GPU_POOL_ACTUATE_RESULT_CHANNEL, BaseEnvelope(
                kind=GPU_ACTUATE_RESULT_KIND, source=ServiceRef(name="gpu2-actuator-fixture"),
                payload=GpuActuateResultV1(action_id=msg.action_id, generation=msg.generation, role=msg.role,
                                           action=msg.action, status=status, **kw).model_dump(mode="json")))

        await answer("accepted")
        gpu.live[SEAT] = LIVE["agent"]
        gpu.down.add("diffusion")
        await answer("succeeded", elapsed_ms=90000, observed={SEAT: "running", "diffusion": "exited"})

    async def result(env):
        await gpu.rt.on_actuate_result(GpuActuateResultV1.model_validate(env.payload))

    bus.handlers[GPU_POOL_ACTUATE_REQUEST_CHANNEL] = actuator
    bus.handlers[GPU_POOL_ACTUATE_RESULT_CHANNEL] = result


@pytest.mark.parametrize("placement", ["home", "gpu2"], ids=["home-agent-card", "gpu2-loaded-by-the-pool"])
@pytest.mark.parametrize("repair_required", [False, True], ids=["accepted-draft", "conditional-repair"])
def test_curiosity_receipt_wait_restart_grant_dispatch_and_completion(monkeypatch, placement, repair_required):
    async def scenario(pool, saver, store):
        bus = TypedBus()
        gpu = InProcessPool(actuate=(SEAT,) if placement == "gpu2" else ())
        settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI=DSN, ORION_BUS_ENABLED=False,
            DURABLE_RUNS_ADMISSION_ENABLED=True, DURABLE_RUNS_TURN_RPC_TIMEOUT_SEC=0.05,
            DURABLE_RUNS_LEASE_HEARTBEAT_SEC=0.1, DURABLE_RUNS_LEASE_SECONDS=90)
        runner = DurableRunner(settings, bus=bus, checkpointer=saver)
        runtime = AdmissionRuntime(settings, runner, pool, store=store)
        monkeypatch.setenv("POSTGRES_URI", DSN)
        main = importlib.import_module("app.main")
        for name, value in {"runner": runner, "admission": runtime, "capacity": None,
                            "rpc_bus": bus, "_settings": settings}.items():
            monkeypatch.setattr(main, name, value)
        bus.handlers[DURABLE_RUN_REQUEST_CHANNEL] = main._handle_request
        bus.handlers[RESOURCE_EVENT_CHANNEL] = main._handle_request
        bus.handlers[GPU_POOL_EVENT_CHANNEL] = main._handle_request
        adapter = build_turn_adapter(monkeypatch, bus, store, repair_required, authority_app=main.app, pool=gpu)
        bus.handlers[CURIOSITY_TURN_REQUEST_CHANNEL] = adapter.handle_turn
        actions: list[GpuActuateV1] = []
        install_gpu2_actuator(bus, gpu, actions)
        await gpu.boot()
        dispatch = cortex_dispatch()

        async def submit(run_id, *, line="investigate", budget=3600):
            # min ctx rides through to the hold (home variant). The gpu2 variant sends none: the
            # 4.3 scheduler's ``fits`` needs a LIVE ctx_per_slot, so a hold with min_ctx_tokens can
            # never justify loading a seat that is not loaded yet (reported in the 4.5 PR; no live
            # producer sets minimum_context_tokens today).
            requirements = {"structured_output": True}
            if placement == "home":
                requirements["minimum_context_tokens"] = 32768
            request = DurableRunRequestV1(run_id=run_id, workflow="curiosity.investigate",
                correlation_id=str(uuid4()), admission={"requirements": requirements},
                brief={"prompt": "Inspect the isolated fixture ledger and state one bounded conclusion.",
                       "session_id": "isolated-durable-acceptance", "line": line,
                       "source_tag": "curiosity_self_inquiry" if line == "self_inquiry" else "curiosity_investigate",
                       "timeout_sec": budget})
            cortex_request = CortexClientRequest(mode="brain", context={
                "messages": [{"role": "user", "content": "curiosity.investigate"}],
                "user_message": "curiosity.investigate", "session_id": request.brief.session_id,
                "metadata": {"durable_run": request.model_dump(mode="json")}})
            reply = await asyncio.wait_for(dispatch(bus=bus, source=ServiceRef(name="orion-cortex-orch"),
                req=cortex_request, correlation_id=request.correlation_id, admission_enabled=True,
                receipt_timeout_sec=1), 2)
            assert reply.ok and reply.status == "accepted", reply.model_dump(mode="json")
            receipt = reply.metadata["durable_run"]
            assert receipt["run_id"] == run_id and receipt["status"] == "waiting_resource"
            assert await store.get_run(run_id), "receipt preceded durable persistence"
            await bus.drain()
            return request

        try:
            # Another run holds the agent card (the only card class agent has today).
            blocker = await gpu.dispatch(GpuLeaseRequestV1(verb="acquire", request_id="acceptance-holder:1",
                kind="hold", holder="durable-runs:acceptance-holder", work_class="agent",
                priority="background", retryable=True))
            assert blocker.status == "granted" and blocker.grant.role == "agent"
            study = await submit("acceptance-self-study", line="self_inquiry", budget=30)
            await runtime._drive(await store.get_run(study.run_id))
            snapshot = await runtime.graph.aget_state(runtime.config(study.run_id))
            assert snapshot.next == ("resource_wait",)
            [hold] = gpu.leases(holder=f"durable-runs:{study.run_id}")
            assert hold["kind"] == "hold" and hold["status"] == "queued"
            assert not runtime.active and not bus.inflight_rpc and not bus.subscriptions
            assert not adapter.stages
            await asyncio.sleep(0.06)  # exceeds the configured legacy RPC budget
            assert (await runtime.status(study.run_id))["status"] == "waiting_resource"
            await runtime.close()

            # A new runtime (restart) recovers the same thread from the real saver.
            restarted = AdmissionRuntime(settings, runner, pool, store=store)
            monkeypatch.setattr(main, "admission", restarted)
            if placement == "gpu2":
                # The run waits past the seat's after_wait_sec (1200 s): the pool loads gpu2.
                await gpu.later(1230, beat=[blocker.lease_id])
                await bus.drain()
                [load] = actions
                assert (load.role, load.action, load.actuator) == (SEAT, "load", "circe")
                await gpu.later(30, beat=[blocker.lease_id])   # discovery confirms the 27B
                await bus.drain()
                assert gpu.events("swap_started") and gpu.events("swapped")
            else:
                await gpu.dispatch(GpuLeaseRequestV1(verb="release", lease_id=blocker.lease_id, outcome="ok"))
                await bus.drain()
            granted = await gpu.lease(hold["lease_id"])
            expected_role = SEAT if placement == "gpu2" else "agent"
            assert granted["status"] == "granted" and granted["role"] == expected_role
            # The pool's granted event reached durable-runs over the bus and woke the run.
            assert study.run_id in restarted._hints and restarted._wake.is_set()
            await restarted.reconcile()
            await asyncio.wait_for(asyncio.gather(*list(restarted.active.values())), 40)
            await bus.drain()
            status = await restarted.status(study.run_id)
            assert status["status"] == "completed", status
            snapshot = await restarted.graph.aget_state(restarted.config(study.run_id))
            assert not snapshot.next
            assert snapshot.values["text"] == (REPAIRED if repair_required else DRAFT)
            expected_stages = ["stance_react", "fcc_primary", "harness_finalize_reflect"]
            if repair_required:
                expected_stages.append("orion_response_repair")
            assert [row["stage"] for row in adapter.stages] == expected_stages
            for row in adapter.stages:
                assert (row["hold_lease_id"], row["hold_generation"]) == (hold["lease_id"], 1)
                assert row["hold_holder"] == f"durable-runs:{study.run_id}"
                assert "lease_id" not in row  # no durable-runs token rode this turn
            # Every model call was an attach under the run's hold (no self-deadlock), on the
            # hold's role -- and the FCC route was the agent route, never the pool role.
            agent_calls = [g for g in adapter.pool_grants if g["work_class"] == "agent"]
            assert agent_calls and all(g["pool_verb"] == "attach" and g["pool_status"] == "granted"
                                       for g in agent_calls)
            children = gpu.leases(hold_lease_id=hold["lease_id"])
            assert children and {c["role"] for c in children} == {expected_role}
            assert not gpu.leases(kind="request", work_class="agent", hold_lease_id=None)
            [harness] = adapter.requests
            assert harness.fcc_model_label == "llamacpp/agent"
            assert SEAT not in harness.model_dump_json(exclude={"gpu_lease"})
            journal = [env for _channel, env in bus.events if env.kind == "journal.entry.write.v1"
                       and env.payload["entry_id"] == "curiosity-self-inquiry:" + study.run_id]
            assert len(journal) == 1 and journal[0].payload["correlation_id"] == study.correlation_id
            attention = [env for _channel, env in bus.events if env.kind == ATTENTION_SCHEMA_KIND
                         and env.payload["entry_id"] == "curiosity-" + study.run_id]
            assert len(attention) == 1 and attention[0].payload["correlation_id"] == study.correlation_id
            await restarted.reconcile()  # duplicate wakeup/outbox flush is harmless
            await bus.drain()
            completion = [env for _channel, env in bus.events if env.kind == DURABLE_RUN_STATE_KIND
                          and env.payload["run_id"] == study.run_id and env.payload["status"] == "completed"]
            assert len(completion) == 1
            assert completion[0].payload["detail"]["line"] == "self_inquiry"
            assert completion[0].payload["correlation_id"] == study.correlation_id
            history = await store.history(study.run_id)
            assert sum(event["event"] == "run.completed" for event in history) == 1
            assigned = next(e for e in history if e["event"] == "run.lane_assigned")
            assert assigned["detail"]["lane"] == expected_role    # Hub's run view: lane = the hold's role
            assert (await gpu.lease(hold["lease_id"]))["status"] == "released"
            assert await legacy_rows(store) == (0, 0)
            assert not bus.inflight_rpc and not bus.subscriptions
            await restarted.close()
        finally:
            await runtime.close()
            await adapter.close()
            await bus.close()
    asyncio.run(with_database(scenario))
