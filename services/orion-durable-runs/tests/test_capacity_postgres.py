"""Physical request/lease arbitration against disposable real Postgres."""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest

from .test_admission_runtime_postgres import DSN, request, runtime, with_database
from orion.durable_admission.capacity import PostgresCapacityStore
from orion.durable_admission.store import SubmissionConflict
from orion.schemas.resource_admission import CapacityAcquireV1, CapacityTokenV1

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def call(**changes):
    return CapacityAcquireV1(request_id=uuid4().hex, correlation_id="capacity-test",
        lane="agent", backend_key="http://test-backend", max_inflight=1, budget_sec=120, **changes)


def token(result):
    return CapacityTokenV1(**{key: result["permit"][key] for key in ("request_id", "permit_id")})


def test_request_and_durable_grant_have_one_atomic_winner():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        capacity = PostgresCapacityStore(store)
        rt.broker.capacity = capacity
        for index in range(8):
            req = request(f"capacity-race-{index}")
            await rt.submit(req)
            acquire = call().model_copy(update={"backend_key": "http://test-backend/"})
            ordinary, grants = await asyncio.gather(capacity.acquire(acquire), rt.broker.tick())
            assert int(ordinary["acquired"]) + len(grants) == 1
            if ordinary["acquired"]:
                assert await store.get_lease(req.run_id) is None
                await capacity.release(token(ordinary))
                grants = await rt.broker.tick()
            assert len(grants) == 1
            assert not (await capacity.acquire(call()))["acquired"]
            await store.finish_projection(req.run_id, "completed", {})
        await rt.close()
    asyncio.run(with_database(scenario))


def test_owner_serialization_duplicate_fencing_and_revoked_thread_occupancy():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        capacity = PostgresCapacityStore(store)
        rt.broker.capacity = capacity
        req = request("capacity-owner")
        await rt.submit(req)
        lease = (await rt.broker.tick())[0]
        owner = call(lease=lease)
        for key, value in (("resource_key", "llm.route.foreign"), ("demand_id", "foreign:turn")):
            forged = owner.model_copy(update={"lease": owner.lease.model_copy(update={key: value})})
            assert (await capacity.acquire(forged))["reason"] == "resource_lease_stale"
        first = await capacity.acquire(owner)
        assert first["acquired"]
        assert (await capacity.acquire(owner))["permit"] == first["permit"]
        assert (await capacity.acquire(call(lease=lease)))["reason"] == "owner_request_active"
        with pytest.raises(SubmissionConflict):
            await capacity.acquire(owner.model_copy(update={"correlation_id": "different"}))
        await store.finish_projection(req.run_id, "cancelled", {})
        # A thread can outlive its workflow lease, but retains physical capacity.
        assert (await capacity.renew(token(first)))["valid"]
        assert (await capacity.acquire(owner))["reason"] == "resource_lease_stale"
        assert (await capacity.acquire(call()))["reason"] == "capacity_full"
        successor = request("capacity-successor")
        await rt.submit(successor)
        assert await rt.broker.tick() == []
        assert (await capacity.release(token(first)))["released"]
        assert not (await capacity.release(token(first)))["released"]
        next_lease = (await rt.broker.tick())[0]
        assert next_lease["run_id"] == successor.run_id
        assert not (await capacity.renew(token(first)))["valid"]
        await rt.close()
    asyncio.run(with_database(scenario))


def test_capacity_limit_expiry_and_stale_ticket_cannot_rearm():
    async def scenario(pool, saver, store):
        now = [datetime(2026, 9, 13, tzinfo=timezone.utc)]
        store.clock = lambda: now[0]
        cap = PostgresCapacityStore(store, ttl_seconds=30)
        requests = [call().model_copy(update={"max_inflight": 2}) for _ in range(3)]
        results = await asyncio.gather(*(cap.acquire(req) for req in requests))
        assert sum(result["acquired"] for result in results) == 2
        first = next(result for result in results if result["acquired"])
        first_req = next(req for req in requests if req.request_id == first["permit"]["request_id"])
        now[0] += timedelta(seconds=20)
        renewed = await cap.renew(token(first))
        assert renewed["valid"]
        now[0] += timedelta(seconds=31)
        assert not (await cap.renew(token(first)))["valid"]
        assert (await cap.acquire(first_req))["reason"] == "request_finished"
        successor = await cap.acquire(call())
        assert successor["acquired"]
        forged = token(first).model_copy(update={"request_id": successor["permit"]["request_id"]})
        assert not (await cap.release(forged))["released"]
        assert len((await cap.snapshot())["active_permits"]) == 1
    asyncio.run(with_database(scenario))


def test_waiting_durable_demand_drains_existing_requests_without_shadow_reservation():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        cap = PostgresCapacityStore(store)
        rt.broker.capacity = cap
        first = await cap.acquire(call().model_copy(update={"max_inflight": 4}))
        await rt.submit(request("capacity-drain"))
        assert await rt.broker.tick() == []
        assert (await cap.acquire(call().model_copy(update={"max_inflight": 4})))["reason"] == "durable_waiting"
        rt.broker.shadow = True
        await rt.broker.tick()
        assert (await cap.acquire(call().model_copy(update={"max_inflight": 4})))["acquired"]
        await cap.release(token(first))
        await rt.close()
    asyncio.run(with_database(scenario))


def test_disabled_driver_ignores_persisted_drain_decisions_but_keeps_active_leases():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        previous = PostgresCapacityStore(store)
        rt.broker.capacity = previous
        first = await previous.acquire(call().model_copy(update={"max_inflight": 4}))
        req = request("capacity-old-drain")
        await rt.submit(req)
        assert await rt.broker.tick() == []
        decision = (await store.get_demand(req.run_id))["decision"]
        assert decision["eligible_backend_keys"] == ["http://test-backend"]
        assert (await previous.acquire(call().model_copy(update={"max_inflight": 4})))["reason"] == "durable_waiting"

        # The new process serves capacity while admission is disabled/shadow.
        restarted = PostgresCapacityStore(store, reserve_waiting=False)
        ordinary = await restarted.acquire(call().model_copy(update={"max_inflight": 4}))
        assert ordinary["acquired"]
        await restarted.release(token(first))
        await restarted.release(token(ordinary))
        assert len(await rt.broker.tick()) == 1
        assert (await restarted.acquire(call()))["reason"] == "durable_lease_active"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_unknown_backend_occupancy_cannot_hoard_drain_priority():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        cap = PostgresCapacityStore(store)
        rt.broker.capacity = cap
        rt.broker.lanes["agent"]["external_busy"] = None
        req = request("capacity-unknown-slots")
        await rt.submit(req)
        assert await rt.broker.tick() == []
        assert (await store.get_demand(req.run_id))["decision"]["eligible_backend_keys"] == []
        ordinary = await cap.acquire(call().model_copy(update={"max_inflight": 4}))
        assert ordinary["acquired"]
        rt.broker.lanes["agent"]["external_busy"] = True
        assert await rt.broker.tick() == []
        assert (await cap.acquire(call().model_copy(update={"max_inflight": 4})))["reason"] == "durable_waiting"
        await cap.release(token(ordinary))
        rt.broker.lanes["agent"]["external_busy"] = False
        assert len(await rt.broker.tick()) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


def test_capacity_disabled_runtime_does_not_require_request_permit_migration():
    async def scenario(pool, saver, store):
        from app.admission_runtime import AdmissionRuntime
        from app.settings import Settings
        from .test_admission_runtime_postgres import Runner

        async with pool.connection() as conn:
            await conn.execute("ALTER TABLE durable_gateway_permits RENAME TO unused_capacity_permits")
        settings = Settings(postgres_uri=DSN, orion_bus_enabled=False,
                            admission_enabled=True, capacity_enabled=False)
        rt = AdmissionRuntime(settings, Runner(saver), pool, store=store)
        assert rt.broker.capacity is None
        rt.broker.lanes = {"agent": {"backend_key": "http://test-backend", "healthy": True,
                                    "configured": True, "capabilities": {}}}
        req = request("capacity-disabled")
        await rt.submit(req)
        await rt.broker.tick()
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_capacity_http_contract_without_cognition_runtime(monkeypatch):
    async def scenario(pool, saver, store):
        monkeypatch.setenv("POSTGRES_URI", DSN)
        from app import main
        monkeypatch.setattr(main, "admission", None)
        monkeypatch.setattr(main, "capacity", PostgresCapacityStore(store))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://authority") as client:
            req = call()
            response = await client.post("/capacity/acquire", json=req.model_dump(mode="json"))
            assert response.status_code == 200 and response.json()["acquired"]
            payload = token(response.json()).model_dump()
            assert (await client.post("/capacity/renew", json=payload)).json()["valid"]
            assert (await client.get("/capacity")).json()["active_permits"]
            assert (await client.get("/admission")).status_code == 503
            assert (await client.post("/capacity/release", json=payload)).json()["released"]
            assert (await client.get("/capacity")).json()["active_permits"] == []
            invalid = await client.post("/capacity/acquire", json={**req.model_dump(mode="json"), "max_inflight": 0})
            assert invalid.status_code == 422
            monkeypatch.setattr(main, "capacity", None)
            assert (await client.post("/capacity/acquire", json=req.model_dump(mode="json"))).status_code == 503
    asyncio.run(with_database(scenario))
