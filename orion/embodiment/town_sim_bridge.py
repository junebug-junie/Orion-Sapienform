"""Trusted town bridge: source checks, RPC results, forecasts and durable outbox.

Only simulation rails are used. The scoring rail is observational and never
feeds memory, training, self-calibration or an autonomy controller.
"""
from __future__ import annotations
import asyncio
import json
import logging
import urllib.request
from uuid import uuid4
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.resilience import publish_with_reconnect, run_subscription
from orion.embodiment import aitown_client
from orion.embodiment.town_mirror import digest_request, weather_request
from orion.schemas.town_sim import TownSimStoredV1, TownSimEventRequestV1, TownSimEventResultV1, TownSimSceneResolvedV1, TownSimPredictionScoreV1

log=logging.getLogger(__name__)
REQUEST='orion:town:sim:event:request'
RESULT='orion:town:sim:event:result'
KINDS={'town.sim.scene.resolved.v1':('orion:town:sim:scene:resolved',TownSimSceneResolvedV1),
       'town.sim.prediction.scored.v1':('orion:town:sim:prediction:scored',TownSimPredictionScoreV1)}
SOURCES={'orion-hub':'juniper','orion-cortex-exec':'orion','orion-embodiment':'mirror'}

class TownSimBridge:
    def __init__(self,worker):
        self.worker=worker;self.bus=worker._bus;self.settings=worker._settings
        self.tasks=[];self._weather_last=0.0

    def start(self):
        self.tasks=[asyncio.create_task(run_subscription(self.bus,REQUEST,self.handle,stop_event=self.worker._stop,name='town-sim-requests')),
                    asyncio.create_task(run_subscription(self.bus,'orion:world_pulse:digest:item',self.mirror,stop_event=self.worker._stop,name='town-mirror')),
                    asyncio.create_task(run_subscription(self.bus,'orion:town:sim:stored',self.stored,stop_event=self.worker._stop,name='town-sim-stored')),
                    asyncio.create_task(self.proposal_loop()),
                    asyncio.create_task(self.poll())]

    async def stop(self):
        for task in self.tasks:task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)

    async def _emit(self,channel,kind,payload,correlation_id=None):
        env=BaseEnvelope(kind=kind,source=self.worker._service_ref(),correlation_id=correlation_id or uuid4(),payload=payload)
        await publish_with_reconnect(self.bus,channel,env,log_label='town_sim')
        return env

    async def inject(self,request:TownSimEventRequestV1,producer:str):
        if SOURCES.get(producer)!=request.source:raise ValueError('producer/source mismatch')
        if request.world_id!=self.worker._world_id:raise ValueError('wrong world')
        event_id=f'{request.source}:{request.correlation_id}'
        event={'id':event_id,'source':request.source,'kind':request.kind,'title':request.title,'effects':[e.model_dump(exclude_none=True) for e in request.effects]}
        if request.quest:
            event['quest']=request.quest.model_dump(exclude_none=True);event['quest']['id']=event_id
        if request.expires_at is not None:event['expires_at']=request.expires_at
        args={'worldId':request.world_id,'event':event,'caps':{'orion':self.settings.town_orion_events_per_day,'mirror':self.settings.town_mirror_per_day,'quests':self.settings.town_orion_open_quests_max}}
        if request.source_ref:args['source_ref']=request.source_ref.model_dump()
        return await asyncio.to_thread(aitown_client.convex_mutation,'sim/runtime:inject',args)

    async def handle(self,raw):
        decoded=self.bus.codec.decode(raw.get('data'))
        if not decoded.ok:return
        env=decoded.envelope
        if env.kind!='town.sim.event.request.v1':return
        try:request=TownSimEventRequestV1.model_validate(env.payload)
        except ValueError:log.warning('invalid town request');return
        try:
            result=await self.inject(request,env.source.name)
            payload=TownSimEventResultV1(correlation_id=request.correlation_id,world_id=request.world_id,accepted=True,reason='accepted',event_id=result['event_id'])
        except Exception as exc:
            payload=TownSimEventResultV1(correlation_id=request.correlation_id,world_id=request.world_id,accepted=False,reason=str(exc)[:300])
        await self._emit(RESULT,'town.sim.event.result.v1',payload.model_dump(),env.correlation_id)
        # Reply only inside the dedicated town RPC namespace.
        if env.reply_to and env.reply_to.startswith('orion:town:sim:reply:'):
            await self._emit(env.reply_to,'town.sim.event.result.v1',payload.model_dump(),env.correlation_id)

    async def mirror(self,raw):
        decoded=self.bus.codec.decode(raw.get('data'))
        if not decoded.ok:return
        env=decoded.envelope
        if env.source.name!='orion-world-pulse':return
        request=digest_request(env.payload,self.worker._world_id)
        if not request:return
        try:
            # Retain provenance without copying headline/body text into the town.
            await asyncio.to_thread(aitown_client.convex_mutation,'sim/runtime:retainSource',{'worldId':request.world_id,'id':request.source_ref.id,'payload':{'kind':'digest','item_id':request.source_ref.id,'category':env.payload['category']}})
            await self.inject(request,'orion-embodiment')
        except Exception:log.info('town mirror skipped/refused',exc_info=True)

    async def forecast(self,row):
        from orion.cognition.plan_loader import build_plan_for_verb
        from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
        from orion.cognition.cortex_payload_extract import extract_cortex_payload_text
        correlation=str(uuid4())
        req=PlanExecutionRequest(plan=build_plan_for_verb('town_forecast',mode='quick'),args=PlanExecutionArgs(request_id=correlation,trigger_source='orion-embodiment',extra={'llm_route':'quick_background'}),context={'user_message':json.dumps(row['request']), 'metadata':{'surface':'aitown','simulation':True}})
        env=BaseEnvelope(kind=req.kind,source=self.worker._service_ref(),correlation_id=correlation,reply_to=f'orion:exec:result:{uuid4()}',payload=req.model_dump(mode='json'))
        result=await self.bus.rpc_request(self.settings.cortex_request_channel,env,reply_channel=env.reply_to,timeout_sec=15)
        # Same codec/text extraction contract as the speech path.
        decoded=self.bus.codec.decode(result.get('data')) if isinstance(result,dict) and 'data' in result else None
        payload=decoded.envelope.payload if decoded and decoded.ok else getattr(result,'payload',result)
        result=payload.get('result') if isinstance(payload,dict) else None
        text=extract_cortex_payload_text(result if isinstance(result,dict) else (payload or {}))
        probabilities=json.loads(text)
        await asyncio.to_thread(aitown_client.convex_mutation,'sim/runtime:submit',{'worldId':self.worker._world_id,'targetId':row['targetId'],'probabilities':probabilities})

    async def propose_event(self,claim):
        from orion.cognition.plan_loader import build_plan_for_verb
        from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
        correlation=str(uuid4())
        req=PlanExecutionRequest(plan=build_plan_for_verb('town_add_event',mode='quick'),args=PlanExecutionArgs(request_id=correlation,trigger_source='orion-embodiment',extra={'llm_route':'quick_background'}),context={'user_message':json.dumps(claim['state']), 'town_world_id':self.worker._world_id,'town_request_id':f"background:{claim['day']}",'metadata':{'surface':'aitown','simulation':True}})
        reply=f'orion:exec:result:{uuid4()}'
        env=BaseEnvelope(kind=req.kind,source=self.worker._service_ref(),correlation_id=correlation,reply_to=reply,payload=req.model_dump(mode='json'))
        await self.bus.rpc_request(self.settings.cortex_request_channel,env,reply_channel=reply,timeout_sec=30)

    async def poll_once(self):
        wid=self.worker._world_id
        if not wid:return
        await asyncio.to_thread(aitown_client.convex_mutation,'sim/runtime:bootstrap',{'worldId':wid})
        pending=await asyncio.to_thread(aitown_client.convex_query,'sim/runtime:pendingOutbox',{'worldId':wid})
        for row in pending or []:
            channel,model=KINDS[row['kind']];payload=model.model_validate(row['payload']).model_dump(mode='json')
            await self._emit(channel,row['kind'],payload)
            # A SQL commit acknowledgement retires this outbox row, not publish success.
        rows=await asyncio.to_thread(aitown_client.convex_query,'sim/runtime:openPredictions',{'worldId':wid})
        for row in rows or []:
            try:await self.forecast(row)
            except Exception:log.info('town forecast missed/refused target=%s',row['targetId'],exc_info=True)

    async def stored(self,raw):
        decoded=self.bus.codec.decode(raw.get('data'))
        if not decoded.ok:return
        env=decoded.envelope
        if env.kind!='town.sim.stored.v1' or env.source.name!='orion-sql-writer':return
        record=TownSimStoredV1.model_validate(env.payload)
        if record.world_id!=self.worker._world_id:return
        await asyncio.to_thread(aitown_client.convex_mutation,'sim/runtime:acknowledge',{'worldId':record.world_id,'eventId':record.event_id,'kind':record.record_kind})

    async def maybe_propose(self):
        claim=await asyncio.to_thread(aitown_client.convex_mutation,'sim/runtime:claimEventProposal',{'worldId':self.worker._world_id})
        if claim:
            try:await self.propose_event(claim)
            except Exception:log.warning('town event proposal failed; no fabricated fallback',exc_info=True)

    async def proposal_loop(self):
        while not self.worker._stop.is_set():
            try:
                if self.worker._world_id:await self.maybe_propose()
            except Exception:log.warning('town proposal bridge unavailable',exc_info=True)
            try:await asyncio.wait_for(self.worker._stop.wait(),timeout=60)
            except asyncio.TimeoutError:pass

    async def poll_weather(self):
        url=self.settings.town_weather_provider_url;region=self.settings.town_weather_region
        if not url or not region:return
        from urllib.parse import urlparse
        if urlparse(url).scheme!='https':raise ValueError('public weather provider requires HTTPS')
        def read():
            with urllib.request.urlopen(url,timeout=5) as response:return json.loads(response.read(65536))
        observed=await asyncio.to_thread(read)
        result=weather_request(observed,self.worker._world_id,url,region)
        if not result:return
        request,snapshot=result
        await asyncio.to_thread(aitown_client.convex_mutation,'sim/runtime:retainSource',{'worldId':request.world_id,'id':request.source_ref.id,'payload':snapshot})
        await self.inject(request,'orion-embodiment')

    async def poll(self):
        while not self.worker._stop.is_set():
            try:
                self.worker._refresh_aitown_runtime()
                await self.poll_once()

                now=asyncio.get_running_loop().time()
                if now-self._weather_last>=3600:
                    self._weather_last=now;await self.poll_weather()
            except Exception:log.warning('town simulation bridge unavailable',exc_info=True)
            try:await asyncio.wait_for(self.worker._stop.wait(),timeout=2)
            except asyncio.TimeoutError:pass
