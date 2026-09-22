"""Isolated real Postgres checkpoints, resource transactions and driver recovery.

Set ORION_ADMISSION_TEST_DSN to a disposable database. Each test uses a fresh
schema; this test never connects to POSTGRES_URI or production configuration.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]
from app.admission_runtime import AdmissionRuntime
from app.graph import Deps
from app.settings import Settings
from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.store import PostgresAdmissionStore
from orion.schemas.durable_run import DurableRunRequestV1, CuriosityTurnResultV1

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


class Runner:
    def __init__(self, saver, block=None):
        self._checkpointer = saver
        self.calls = []
        self.events = []
        self.block = block
        self._bus = None

    def _curiosity_deps(self):
        async def turn(req):
            self.calls.append(req)
            if self.block:
                await self.block.wait()
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="Evidence-backed finding")
        async def read(run_id):
            return {"graph_readable": True}
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

    async def _publish(self, channel, kind, model, corr):
        self.events.append((kind, model))
        return True

    async def _emit_state(self, *args, **kwargs):
        self.events.append(("state", kwargs))

    def _corr_for_admission(self, value):
        return value


def request(run_id):
    return DurableRunRequestV1(run_id=run_id, workflow="curiosity.investigate", correlation_id=str(uuid4()),
        admission={}, brief={"prompt": "Inspect the current self-inquiry evidence.", "session_id": "curiosity", "timeout_sec": 100})


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


def runtime(pool, saver, store, block=None):
    settings = Settings(postgres_uri=DSN, orion_bus_enabled=False, admission_enabled=True,
                        lease_heartbeat_sec=0.03, retry_base_sec=0.05)
    runner = Runner(saver, block)
    broker = ResourceBroker(store, lanes={"agent": {"backend_key": "http://test-backend", "healthy": True,
                            "configured": True, "capabilities": {}}}, lease_seconds=90)
    return AdmissionRuntime(settings, runner, pool, store=store, broker=broker)


def test_persisted_acceptance_wait_restart_complete_and_terminal_duplicate():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        holder = request("holder-001")
        await store.submit(holder.model_dump(mode="json"))
        await store.register_demand(holder.run_id, holder.admission.model_dump(mode="json"))
        await rt.broker.tick()
        waiting = request("waiting-001")
        receipt = await asyncio.wait_for(rt.submit(waiting), 1)
        assert receipt["status"] == "waiting_resource"
        await rt._drive(await store.get_run(waiting.run_id))
        assert not rt.active and not rt.runner.calls
        assert (await rt.graph.aget_state(rt.config(waiting.run_id))).next == ("resource_wait",)
        await rt.close()
        restarted = runtime(pool, saver, store)
        await store.release(await store.get_lease(holder.run_id), "holder_completed")
        await restarted.broker.tick()
        await restarted._drive(await store.get_run(waiting.run_id))
        assert len(restarted.runner.calls) == 1
        assert (await store.get_run(waiting.run_id))["terminal"] == "completed"
        assert await store.get_lease(waiting.run_id) is None
        assert (await restarted.submit(waiting))["status"] == "completed"
        await restarted.reconcile()  # durable event outbox can be replayed
        await restarted.close()
        events = await store.history(waiting.run_id)
        assert any(e["event"] == "run.completed" for e in events)
    asyncio.run(with_database(scenario))


def test_cancel_running_fences_token_and_releases_capacity():
    async def scenario(pool, saver, store):
        block = asyncio.Event()
        rt = runtime(pool, saver, store, block)
        req = request("cancel-001")
        await rt.submit(req)
        await rt.reconcile()
        for _ in range(100):
            if rt.runner.calls:
                break
            await asyncio.sleep(0.01)
        assert rt.runner.calls
        assert (await rt.status(req.run_id))["status"] == "running"
        token = await store.get_lease(req.run_id)
        await rt.control(req.run_id, "cancel")
        assert (await rt.control(req.run_id, "resume"))["status"] == "cancelled"
        assert not await store.validate(token)
        await rt._drive(await store.get_run(req.run_id))
        assert (await rt.status(req.run_id))["status"] == "cancelled"
        assert not (await rt.graph.aget_state(rt.config(req.run_id))).next
        await rt.close()
    asyncio.run(with_database(scenario))


def test_curiosity_candidates_use_explicit_policy_and_stay_stable_on_duplicate():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        rt.settings.lane_policy_json = json.dumps({
            "metacog": {"compatible_with": ["agent"]},
            "chat": {"compatible_with": []},
        })
        req = request("policy-001")  # Same minimal demand as the Hub producer.
        await rt.submit(req)
        original = await store.get_run(req.run_id)
        assert original["request"]["admission"]["alternatives"] == ["metacog"]
        assert req.admission.alternatives == []
        rt.settings.lane_policy_json = "{}"
        await rt.submit(req)
        assert (await store.get_run(req.run_id))["request"] == original["request"]
        # An explicit declaration never fabricates a route in the live catalog.
        assert "metacog" not in rt.broker.lanes
        await rt.close()
    asyncio.run(with_database(scenario))


def test_duplicate_submission_does_not_rearm_backoff_and_aliases_share_one_winner():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("backoff-001")
        await rt.submit(req)
        await store.suspend_demand(req.run_id)
        await rt.submit(req)
        assert (await store.get_demand(req.run_id))["status"] == "suspended"
        # Two logical routes to the same canonical URL cannot grant twice.
        await store.register_demand(req.run_id, req.admission.model_dump(mode="json"))
        other = request("alias-001").model_copy(update={"admission": req.admission.model_copy(update={"preferred_lane": "metacog", "resource": "llm.route.metacog"})})
        await rt.submit(other)
        rt.broker.lanes["metacog"] = {**rt.broker.lanes["agent"], "backend_key": "http://test-backend/"}
        grants = await asyncio.gather(rt.broker.tick(), rt.broker.tick())
        assert sum(len(g) for g in grants) == 1
        assert await store.get_lease(req.run_id)
        assert await store.get_lease(other.run_id) is None
        await rt.close()
    asyncio.run(with_database(scenario))


def test_renew_expiry_fencing_and_fake_clock_widening():
    async def scenario(pool, saver, store):
        now = [datetime(2026, 9, 12, tzinfo=timezone.utc)]
        store.clock = lambda: now[0]
        rt = runtime(pool, saver, store)
        rt.broker.widening_enabled = True
        rt.broker.lanes["metacog"] = {"backend_key": "http://alternative", "configured": True,
                                      "healthy": True, "compatible_with": ["agent"], "capabilities": {}}
        holder = request("holder-001").model_copy(update={"brief": request("unused-001").brief.model_copy(update={"timeout_sec": 3500})})
        waiter = request("waiter-001")
        waiter.admission.alternatives = ["metacog"]
        await rt.submit(holder)
        token = (await rt.broker.tick())[0]
        await rt.submit(waiter)
        assert await rt.broker.tick() == []
        initial_age = (await store.get_demand(waiter.run_id))["created_at"]
        for _ in range(16):
            now[0] += timedelta(seconds=75)
            assert await store.renew(token, 90)
        grants = await rt.broker.tick()
        assert len(grants) == 1 and grants[0]["lane"] == "metacog"
        demand = await store.get_demand(waiter.run_id)
        assert demand["created_at"] == initial_age
        assert demand["decision"]["eligible_lanes"] == ["agent", "metacog"]
        await store.release(grants[0], "completed")
        assert not await store.release(grants[0], "duplicate")
        now[0] += timedelta(seconds=91)
        assert len(await store.expire()) == 1 and await store.expire() == []
        assert not await store.validate(token) and await store.renew(token, 90) is None
        await store.register_demand(holder.run_id, holder.admission.model_dump(mode="json"))
        newer = (await rt.broker.tick())[0]
        assert newer["generation"] > token["generation"]
        assert not await store.release(token, "stale_release") and await store.validate(newer)
    asyncio.run(with_database(scenario))
