"""Actual SQL ownership races; fake clock and physical controller only."""
import asyncio
from datetime import datetime,timedelta,timezone
from unittest.mock import AsyncMock
import pytest
from .test_admission_runtime_postgres import DSN,request,runtime,with_database
from .test_capacity_postgres import call,token
from orion.durable_admission.elastic import ElasticStore
from orion.durable_admission.capacity import PostgresCapacityStore
from app.elastic_runtime import ElasticRuntime

pytestmark=pytest.mark.skipif(not DSN,reason="isolated ORION_ADMISSION_TEST_DSN required")
BACKEND="http://burst"

def burst_request(name):
    r=request(name)
    return r.model_copy(update={"admission":r.admission.model_copy(update={"allow_elastic_activation":True,"alternatives":["agent-burst"]})})

def lanes():
    return {"agent":{"configured":True,"healthy":True,"backend_key":"http://test-backend","external_busy":True,"busy_budget_seconds":3600,"capabilities":{}},
        "agent-burst":{"configured":True,"healthy":False,"backend_key":BACKEND,"external_busy":None,
        "activatable":True,"compatible_with":["agent"],"activation_capabilities":{},"capabilities":{}}}

async def setup(pool,saver,store):
    rt=runtime(pool,saver,store)
    rt.broker.lanes=lanes();rt.broker.widening_enabled=True
    rt.broker.elastic=ElasticStore(store,BACKEND)
    await rt.broker.elastic.initialize()
    rt.broker.elastic_environment={"eligible":True}
    rt.broker.elastic_shadow=False
    rt.broker.elastic_budget={"drain":300,"transition":60,"cold":600}
    return rt

def test_threshold_intent_restart_readiness_fifo_and_no_attempt():
    async def scenario(pool,saver,store):
        now=datetime(2026,9,14,tzinfo=timezone.utc);store.clock=lambda:now
        rt=await setup(pool,saver,store)
        await rt.submit(burst_request("older-run"));now+=timedelta(seconds=1)
        await rt.submit(burst_request("younger"))
        await rt.broker.tick()
        assert (await rt.broker.elastic.snapshot())["operation_id"] is None
        now+=timedelta(seconds=1200)
        assert await rt.broker.tick() == []
        row=await rt.broker.elastic.snapshot()
        assert row["run_id"] == "older-run" and row["state"] == "requested"
        recovered=ElasticStore(store,BACKEND)
        assert (await recovered.snapshot())["operation_id"] == row["operation_id"]
        assert not rt.runner.calls
        await recovered.complete(row["operation_id"],{"status":"success"},healthy=False,assignments=True)
        assert await rt.broker.tick() == []
        await recovered.complete(row["operation_id"],{"status":"success"},healthy=True,assignments=True)
        rt.broker.lanes=lanes();rt.broker.lanes["agent-burst"].update(healthy=True,external_busy=False)
        grants=await rt.broker.tick()
        assert len(grants)==1 and grants[0]["run_id"] == "older-run"
        assert (await store.get_demand("older-run"))["created_at"] == now-timedelta(seconds=1201)
        assert not rt.runner.calls
        await rt.close()
    asyncio.run(with_database(scenario))

def test_closure_preserves_owner_calls_and_rejects_aliases():
    async def scenario(pool,saver,store):
        now=datetime(2026,9,14,tzinfo=timezone.utc);store.clock=lambda:now
        rt=await setup(pool,saver,store);capacity=PostgresCapacityStore(store)
        await rt.submit(burst_request("owner-run"));now+=timedelta(seconds=1200)
        await rt.broker.tick();row=await rt.broker.elastic.snapshot()
        await rt.broker.elastic.complete(row["operation_id"],{"status":"success"},healthy=True,assignments=True)
        rt.broker.lanes=lanes();rt.broker.lanes["agent-burst"].update(healthy=True,external_busy=False)
        lease=(await rt.broker.tick())[0]
        async with store.transaction() as conn:
            await rt.broker.elastic.intent(conn,"diffusion","owner-run",{})
        owner=call(lease=lease).model_copy(update={"lane":"agent-burst","backend_key":BACKEND})
        first=await capacity.acquire(owner)
        assert first["acquired"]  # full sequential owner chain survives closure
        assert not (await rt.broker.elastic.snapshot())["can_transition"]
        assert not (await capacity.acquire(call().model_copy(update={"lane":"alias","backend_key":BACKEND})))["acquired"]
        await store.release(lease,"finished")
        assert (await capacity.renew(token(first)))["valid"]
        assert not (await rt.broker.elastic.snapshot())["can_transition"]
        await capacity.release(token(first))
        assert (await rt.broker.elastic.snapshot())["can_transition"]
        await rt.close()
    asyncio.run(with_database(scenario))

def test_first_ready_reconcile_does_not_close_before_grant(monkeypatch):
    async def scenario(pool,saver,store):
        now=datetime.now(timezone.utc);store.clock=lambda:now
        rt=await setup(pool,saver,store)
        rt.settings.elastic_backend=BACKEND;rt.settings.elastic_assignments=True
        rt.settings.elastic_shadow=False;rt.settings.capacity_enabled=True
        rt.settings.elastic_idle_grace=0
        await rt.submit(burst_request("first-run"));now+=timedelta(seconds=1200)
        await rt.broker.tick();row=await rt.broker.elastic.snapshot()
        await rt.broker.elastic.complete(row["operation_id"],{"status":"success"},healthy=True,assignments=True)
        rt.broker.lanes=lanes();rt.broker.lanes["agent-burst"].update(healthy=True,external_busy=False)
        elastic=ElasticRuntime(rt);elastic.initialized=True
        monkeypatch.setattr(elastic,"environment",AsyncMock(return_value={"eligible":True}))
        await elastic.tick()
        assert (await elastic.store.snapshot())["admissions_open"]
        assert len(await rt.broker.tick()) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


def test_maximum_borrow_closes_new_leases_but_does_not_stop_owner(monkeypatch):
    async def scenario(pool,saver,store):
        now=datetime.now(timezone.utc);store.clock=lambda:now
        rt=await setup(pool,saver,store)
        rt.settings.elastic_backend=BACKEND;rt.settings.elastic_assignments=True
        rt.settings.elastic_shadow=False;rt.settings.elastic_restoration=True
        rt.settings.capacity_enabled=True;rt.settings.elastic_max_borrow=1300
        await rt.submit(burst_request("maximum-owner"));now+=timedelta(seconds=1200)
        await rt.broker.tick();row=await rt.broker.elastic.snapshot()
        await rt.broker.elastic.complete(row["operation_id"],{"status":"success"},healthy=True,assignments=True)
        rt.broker.lanes=lanes();rt.broker.lanes["agent-burst"].update(healthy=True,external_busy=False)
        lease=(await rt.broker.tick())[0]
        # Renewing owner represents a real long attempt, not an expired lease.
        now+=timedelta(seconds=1301)
        async with store.transaction() as conn:
            await conn.execute("UPDATE durable_resource_leases SET expires_at=%s WHERE lease_id=%s",(now+timedelta(seconds=90),lease['lease_id']))
        elastic=ElasticRuntime(rt);elastic.initialized=True
        monkeypatch.setattr(elastic,"environment",AsyncMock(return_value={"eligible":True}))
        await elastic.tick()
        closed=await elastic.store.snapshot()
        assert not closed['admissions_open'] and not closed['can_transition']
        assert elastic.job is None
        monkeypatch.setattr(elastic,"actuate",AsyncMock())
        await store.release(lease,'finished')
        # Same tick requests restoration only once owner is gone.
        await elastic.tick()
        assert (await elastic.store.snapshot())['desired_target']=='diffusion'
        await elastic.close();await rt.close()
    asyncio.run(with_database(scenario))


def test_failed_activation_records_restored_diffusion_residency(monkeypatch):
    async def scenario(pool,saver,store):
        now=datetime.now(timezone.utc);store.clock=lambda:now
        rt=await setup(pool,saver,store)
        rt.settings.elastic_backend=BACKEND;rt.settings.capacity_enabled=True
        await rt.submit(burst_request('rollback-test'));now+=timedelta(seconds=1200)
        await rt.broker.tick();row=await rt.broker.elastic.snapshot()
        await rt.broker.elastic.complete(row['operation_id'],{'status':'failed','restored':True},healthy=False,assignments=True)
        restored=await rt.broker.elastic.snapshot()
        assert restored['state']=='idle' and restored['desired_target']=='diffusion'
        assert restored['last_restored_at']==now
        elastic=ElasticRuntime(rt);elastic.initialized=True
        monkeypatch.setattr(elastic,'environment',AsyncMock(return_value={'eligible':True}))
        for _ in range(3):
            await elastic.tick();await rt.broker.tick()
            assert (await elastic.store.snapshot())['generation']==row['generation']
        await rt.close()
    asyncio.run(with_database(scenario))


def test_diffusion_restore_does_not_depend_on_gateway(monkeypatch):
    import httpx
    async def scenario(pool,saver,store):
        rt=await setup(pool,saver,store)
        rt.settings.elastic_backend=BACKEND
        await rt.submit(burst_request('restore-gateway-offline'))
        elastic=ElasticRuntime(rt);elastic.initialized=True
        async with store.transaction() as conn:
            await elastic.store.intent(conn,'diffusion','restore-gateway-offline',{})
        gateway=AsyncMock(side_effect=RuntimeError('gateway offline'))
        monkeypatch.setattr(rt,'refresh_lanes',gateway)
        client_type=httpx.AsyncClient
        transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'status':'success','state':'ready'}))
        monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:client_type(transport=transport,**kw))
        await elastic.actuate()
        result=await elastic.store.snapshot()
        assert result['state']=='idle' and result['last_restored_at'] is not None
        gateway.assert_not_called()
        await rt.close()
    asyncio.run(with_database(scenario))
