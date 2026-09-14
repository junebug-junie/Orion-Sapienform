"""Finite contention eval: inactive GPU2, durable intent, FIFO, restoration.

Only disposable ORION_ADMISSION_TEST_DSN; physical facts are labelled fixtures.
"""
import asyncio
import json
import os
from datetime import datetime,timedelta,timezone
from uuid import uuid4
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.store import PostgresAdmissionStore
from orion.durable_admission.elastic import ElasticStore
from orion.schemas.durable_run import DurableRunRequestV1

async def main():
    dsn=os.environ['ORION_ADMISSION_TEST_DSN'];schema='elastic_eval_'+uuid4().hex
    async with await AsyncConnection.connect(dsn,autocommit=True) as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
    now=datetime(2026,9,14,tzinfo=timezone.utc)
    async with AsyncConnectionPool(dsn,min_size=1,max_size=4,open=False,kwargs={"autocommit":True,"row_factory":dict_row,"options":f'-c search_path={schema},public'}) as pool:
        store=PostgresAdmissionStore(pool,clock=lambda:now);await store.setup()
        elastic=ElasticStore(store,'http://fixture-burst');await elastic.initialize()
        broker=ResourceBroker(store,{},widening_enabled=True);broker.elastic=elastic
        broker.elastic_shadow=False;broker.elastic_environment={"eligible":True,"source":"fixture"}
        broker.elastic_budget={"drain":300,"transition":60,"cold":600}
        def lanes(ready):
            return {"agent":{"backend_key":"http://fixture-agent","configured":True,"healthy":True,"external_busy":True,"capabilities":{}},
                "agent-burst":{"backend_key":"http://fixture-burst","configured":True,"healthy":ready,"external_busy":False if ready else None,"activatable":True,"compatible_with":["agent"],"capabilities":{},"activation_capabilities":{}}}
        for i in range(20):
            request=DurableRunRequestV1(run_id=f'eval-run-{i:02}',workflow='curiosity.investigate',correlation_id=str(uuid4()),brief={"prompt":"fixture only","session_id":"eval","timeout_sec":100},admission={"allow_elastic_activation":True,"alternatives":["agent-burst"]})
            await store.submit(request.model_dump(mode='json'));await store.register_demand(request.run_id,request.admission.model_dump(mode='json'))
            now+=timedelta(seconds=1)
        broker.lanes=lanes(False);assert not await broker.tick()
        assert (await elastic.snapshot())["operation_id"] is None
        now+=timedelta(seconds=1200);assert not await broker.tick()
        intent=await elastic.snapshot();assert intent['run_id']=='eval-run-00'
        await elastic.complete(intent['operation_id'],{'status':'success'},healthy=True,assignments=True)
        served=[]
        for i in range(20):
            broker.lanes=lanes(True);grants=await broker.tick()
            assert len(grants)==1
            served.append(grants[0]['run_id'])
            await store.finish_projection(grants[0]['run_id'],'completed',{})
        assert served==[f'eval-run-{i:02}' for i in range(20)]
        async with store.transaction() as conn:
            restored=await elastic.intent(conn,'diffusion',served[-1],{'reason':'eval_queue_drained'})
        assert (await elastic.snapshot())['can_transition']
        await elastic.complete(restored['operation_id'],{'status':'success'},healthy=True,assignments=False)
        assert (await elastic.snapshot())['state']=='idle'
        print(json.dumps({'result':'passed','served':len(served),'fifo_order':served,'inactive_lane_fixture':True,'restored':True,'production_exercised':False}))
if __name__=='__main__':asyncio.run(main())
