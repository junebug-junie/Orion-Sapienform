"""World-model / visual-chain GPU permits (``/capacity``) against disposable real Postgres.

Stage 4.5: durable runs no longer take durable leases (the GPU pool holds them), so the permit
store's durable-lease branches only ever see FROZEN legacy rows. These permits themselves stay until
stage 5 moves world-model and diffusion onto pool leases."""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest

from psycopg.types.json import Jsonb

from .test_admission_runtime_postgres import DSN, request, runtime, with_database
from orion.durable_admission.capacity import PostgresCapacityStore
from orion.schemas.resource_admission import CapacityAcquireV1, CapacityTokenV1

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def call(**changes):
    return CapacityAcquireV1(request_id=uuid4().hex, correlation_id="capacity-test",
        lane="agent", backend_key="http://test-backend", max_inflight=1, budget_sec=120, **changes)


def token(result):
    return CapacityTokenV1(**{key: result["permit"][key] for key in ("request_id", "permit_id")})


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


def test_capacity_disabled_runtime_does_not_require_request_permit_migration():
    async def scenario(pool, saver, store):
        async with pool.connection() as conn:
            await conn.execute("ALTER TABLE durable_gateway_permits RENAME TO unused_capacity_permits")
        rt = runtime(pool, saver, store, DURABLE_RUNS_CAPACITY_ENABLED=False)
        req = request("capacity-disabled")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.close()
    asyncio.run(with_database(scenario))


async def _seed_frozen_demand(store, run_id, backend):
    """A pre-cutover pending demand whose last broker decision reserved ``backend``."""
    req = request(run_id)
    await store.submit(req.model_dump(mode="json"))
    async with store.pool.connection() as conn:
        await conn.execute(
            "INSERT INTO durable_resource_demands(demand_id,run_id,requirement,status,decision) "
            "VALUES (%s,%s,%s,'pending',%s)",
            (f"{run_id}:harness_turn:llm.route.agent", run_id, Jsonb(req.admission.model_dump(mode="json")),
             Jsonb({"eligible_backend_keys": [backend]})))


def test_frozen_pending_demands_no_longer_reserve_capacity(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", DSN)
    """The 13 frozen demands must not drain world/visual permits: no broker exists to grant them
    (main.py builds the permit store with reserve_waiting=False since 4.5)."""
    async def scenario(pool, saver, store):
        await _seed_frozen_demand(store, "frozen-001", "http://test-backend")
        legacy = PostgresCapacityStore(store)   # the pre-4.5 wiring
        assert (await legacy.acquire(call().model_copy(update={"max_inflight": 4})))["reason"] == "durable_waiting"
        current = PostgresCapacityStore(store, reserve_waiting=False)
        permit = await current.acquire(call().model_copy(update={"max_inflight": 4}))
        assert permit["acquired"]
        await current.release(token(permit))
        import inspect
        from app import main
        assert "reserve_waiting=False" in inspect.getsource(main.lifespan)
    asyncio.run(with_database(scenario))


def test_an_active_legacy_lease_still_fences_its_backend_until_it_ends():
    """Why the cutover runbook waits for zero active durable_resource_leases first: the permit store
    still honours one, and the pool cannot see it."""
    async def scenario(pool, saver, store):
        await _seed_frozen_demand(store, "legacy-holder", "http://test-backend")
        async with store.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO durable_resource_leases(lease_id,demand_id,run_id,resource_key,lane,backend_key,"
                "granted_at,expires_at,heartbeat_at,status) VALUES ('legacy-lease',%s,'legacy-holder',"
                "'llm.route.agent','agent','http://test-backend',now(),now()+interval '1 hour',now(),'active')",
                ("legacy-holder:harness_turn:llm.route.agent",))
        cap = PostgresCapacityStore(store, reserve_waiting=False)
        assert (await cap.acquire(call()))["reason"] == "durable_lease_active"
        async with store.pool.connection() as conn:
            await conn.execute("UPDATE durable_resource_leases SET status='released'")
        assert (await cap.acquire(call()))["acquired"]
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
            # Deleted with the broker in 4.5 (kill means kill): no stale authority answers.
            for method, path in (("get", "/admission"), ("post", "/leases/validate"),
                                 ("get", "/elastic/status"), ("post", "/elastic/target")):
                assert (await getattr(client, method)(path)).status_code in (404, 405), path
            assert (await client.post("/capacity/release", json=payload)).json()["released"]
            assert (await client.get("/capacity")).json()["active_permits"] == []
            invalid = await client.post("/capacity/acquire", json={**req.model_dump(mode="json"), "max_inflight": 0})
            assert invalid.status_code == 422
            monkeypatch.setattr(main, "capacity", None)
            assert (await client.post("/capacity/acquire", json=req.model_dump(mode="json"))).status_code == 503
    asyncio.run(with_database(scenario))
