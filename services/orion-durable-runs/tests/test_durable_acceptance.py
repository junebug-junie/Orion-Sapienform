"""Connected receipt-to-artifact acceptance with real service adapters and Postgres."""
import asyncio
from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.admission_runtime import AdmissionRuntime
from app.runner import DurableRunner
from app.settings import Settings
from orion.core.bus.bus_schemas import ServiceRef
from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.capacity import PostgresCapacityStore
from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.durable_run import (
    DurableRunRequestV1, CURIOSITY_TURN_REQUEST_CHANNEL, DURABLE_RUN_REQUEST_CHANNEL,
    DURABLE_RUN_STATE_KIND,
)
from orion.schemas.resource_admission import RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, ResourceEventV1
from orion.schemas.attention_schema import ATTENTION_SCHEMA_KIND
from .acceptance_bus import TypedBus
from .acceptance_turn import build_turn_adapter, DRAFT, REPAIRED
from .test_admission_runtime_postgres import DSN, with_database

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
ROOT = Path(__file__).resolve().parents[3]


def cortex_dispatch():
    spec = importlib.util.spec_from_file_location("acceptance_cortex_dispatch",
        ROOT / "services/orion-cortex-orch/app/durable_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dispatch_durable_run


@pytest.mark.parametrize("widen", [False, True, "elastic"], ids=["preferred", "widened", "elastic-burst"])
@pytest.mark.parametrize("repair_required", [False, True], ids=["accepted-draft", "conditional-repair"])
def test_curiosity_receipt_wait_restart_dispatch_and_completion(monkeypatch, widen, repair_required):
    async def scenario(pool, saver, store):
        alternate = "agent-burst" if widen == "elastic" else "metacog"
        bus = TypedBus()
        settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI=DSN, ORION_BUS_ENABLED=False,
            DURABLE_RUNS_ADMISSION_ENABLED=True, DURABLE_RUNS_ADMISSION_SHADOW=False,
            DURABLE_RUNS_CAPACITY_ENABLED=True, DURABLE_RUNS_TURN_RPC_TIMEOUT_SEC=0.05,
            DURABLE_RUNS_LEASE_HEARTBEAT_SEC=0.1, DURABLE_RUNS_LEASE_SECONDS=90,
            DURABLE_RUNS_WIDENING_ENABLED=True, DURABLE_RUNS_WIDENING_AFTER_SEC=1200,
            DURABLE_RUNS_LANE_POLICY_JSON=json.dumps({alternate: {"compatible_with": ["agent"]}}))
        capacity = PostgresCapacityStore(store)
        lanes = {lane: {"backend_key": backend, "configured": True, "healthy": True,
                       "capabilities": {"structured_output": True, "context_tokens": 32768},
                       "compatible_with": ["agent"] if lane == alternate else []}
                 for lane, backend in (("agent", "http://fixture-backend"), (alternate, "http://fixture-metacog"))}
        broker = ResourceBroker(store, lanes, lease_seconds=90, widening_enabled=True,
                                widen_after_seconds=1200, hysteresis_seconds=120, capacity=capacity)
        if widen == "elastic":
            from orion.durable_admission.elastic import ElasticStore
            broker.elastic = ElasticStore(store,"http://fixture-metacog")
            await broker.elastic.initialize()
            broker.elastic_shadow = False
            broker.elastic_environment = {"eligible":True}
            broker.elastic_budget = {"drain":300,"transition":60,"cold":600}
            lanes[alternate].update(healthy=False,activatable=True,
                activation_capabilities={"structured_output":True,"context_tokens":32768})
        runner = DurableRunner(settings, bus=bus, checkpointer=saver)
        runtime = AdmissionRuntime(settings, runner, pool, store=store, broker=broker)
        monkeypatch.setenv("POSTGRES_URI", DSN)
        main = importlib.import_module("app.main")
        for name, value in {"runner": runner, "admission": runtime, "capacity": capacity,
                            "rpc_bus": bus, "_settings": settings}.items():
            monkeypatch.setattr(main, name, value)
        bus.handlers[DURABLE_RUN_REQUEST_CHANNEL] = main._handle_request
        bus.handlers[RESOURCE_EVENT_CHANNEL] = main._handle_request
        adapter = build_turn_adapter(monkeypatch, bus, capacity, store, repair_required, authority_app=main.app)
        bus.handlers[CURIOSITY_TURN_REQUEST_CHANNEL] = adapter.handle_turn
        dispatch = cortex_dispatch()

        async def submit(run_id, *, line="investigate", budget=3600):
            request = DurableRunRequestV1(run_id=run_id, workflow="curiosity.investigate",
                correlation_id=str(uuid4()), admission={"allow_elastic_activation": widen == "elastic", "requirements": {
                    "structured_output": True, "minimum_context_tokens": 32768}},
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
            holder = await submit("acceptance-holder")
            assert len(await broker.tick()) == 1
            holder_lease = await store.get_lease(holder.run_id)
            # Model the continuing holder's heartbeat across the clock jump.
            assert await store.renew(holder_lease, 3600)
            study = await submit("acceptance-self-study", line="self_inquiry", budget=30)
            await runtime._drive(await store.get_run(study.run_id))
            snapshot = await runtime.graph.aget_state(runtime.config(study.run_id))
            assert snapshot.next == ("resource_wait",)
            assert not runtime.active and not bus.inflight_rpc and not bus.subscriptions
            assert not adapter.stages
            await asyncio.sleep(0.06)  # exceeds the configured legacy RPC budget
            assert (await runtime.status(study.run_id))["status"] == "waiting_resource"
            await runtime.close()

            # New runtime and real saver connection state recover the same thread.
            restarted = AdmissionRuntime(settings, runner, pool, store=store, broker=broker)
            monkeypatch.setattr(main, "admission", restarted)
            if widen:
                now = datetime.now(timezone.utc) + timedelta(seconds=1201)
                async def advanced_now(_conn):
                    return now
                monkeypatch.setattr(store, "now", advanced_now)
                restarted.now = lambda: now
            else:
                await store.finish_projection(holder.run_id, "completed", {})
            if widen == "elastic":
                assert await broker.tick() == []
                intent=await broker.elastic.snapshot()
                assert intent["state"] == "requested" and intent["run_id"] == study.run_id
                assert not adapter.stages and not bus.inflight_rpc
                # Physical model is a fixture; SQL/FCC/Gateway fencing below is real.
                await broker.elastic.complete(intent["operation_id"],{"status":"success"},healthy=True,assignments=True)
                broker.lanes[alternate]["healthy"]=True
            grants = await broker.tick()
            assert len(grants) == 1
            granted = grants[0]
            expected_lane = alternate if widen else "agent"
            assert granted["lane"] == expected_lane
            if widen:
                decision = (await store.get_demand(study.run_id))["decision"]
                assert decision["eligible_lanes"] == ["agent", alternate]
                assert await store.get_lease(holder.run_id), "widening stole the occupied preferred lane"
            # Deliver the persisted grant through the production handler before
            # reconciliation. Leave it unacked to exercise duplicate delivery too.
            event = ResourceEventV1.model_validate(next(raw for raw in await store.pending_outbox()
                if raw["run_id"] == study.run_id and raw["event"] == "run.resource_granted"))
            restarted._wake.clear()
            assert await runner._publish(RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, event,
                runner._corr_for_admission(event.correlation_id))
            await bus.drain()
            assert restarted._wake.is_set() and not adapter.stages
            # The holder ends after the assignment; the widened run keeps metacog.
            await store.finish_projection(holder.run_id, "completed", {})
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
                assert row["run_id"] == study.run_id and row["lane"] == expected_lane
                assert row["lease_id"] == granted["lease_id"] and row["generation"] == granted["generation"]
            assert adapter.runs[0].response_repair_ran is repair_required
            await store.finish_projection(holder.run_id, "completed", {})
            await restarted.reconcile()
            await bus.drain()
            await restarted.reconcile()  # duplicate wakeup/outbox flush is harmless
            await bus.drain()
            journal = [env for _channel, env in bus.events if env.kind == "journal.entry.write.v1"
                       and env.payload["entry_id"] == "curiosity-self-inquiry:" + study.run_id]
            assert len(journal) == 1 and journal[0].payload["correlation_id"] == study.correlation_id
            attention = [env for _channel, env in bus.events if env.kind == ATTENTION_SCHEMA_KIND
                         and env.payload["entry_id"] == "curiosity-" + study.run_id]
            assert len(attention) == 1 and attention[0].payload["correlation_id"] == study.correlation_id
            completion = [env for _channel, env in bus.events if env.kind == DURABLE_RUN_STATE_KIND
                          and env.payload["run_id"] == study.run_id and env.payload["status"] == "completed"]
            assert len(completion) == 1
            assert completion[0].payload["detail"]["line"] == "self_inquiry"
            assert completion[0].payload["correlation_id"] == study.correlation_id
            history = await store.history(study.run_id)
            assert sum(event["event"] == "run.completed" for event in history) == 1
            assert not await store.get_lease(study.run_id)
            assert not (await capacity.snapshot())["active_permits"]
            assert not bus.inflight_rpc and not bus.subscriptions
            await restarted.close()
        finally:
            await runtime.close()
            await adapter.close()
            await bus.close()
    asyncio.run(with_database(scenario))
