from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime,timezone
import pytest
from orion.core.bus.bus_schemas import BaseEnvelope,ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.embodiment.town_sim_bridge import TownSimBridge
from orion.embodiment.town_mirror import digest_request,weather_request
from orion.schemas.town_sim import TownSimEventRequestV1


def request(**kwargs):
    return TownSimEventRequestV1(correlation_id='req-1',world_id='world',source=kwargs.pop('source','juniper'),kind='delivery',title='Parts arrived',effects=[{'op':'stock_delta','target':'workshop.parts','value':3}],requested_by='operator',**kwargs)


def bridge():
    worker=SimpleNamespace(_world_id='world',_bus=SimpleNamespace(codec=OrionCodec()),_settings=SimpleNamespace(town_orion_events_per_day=5,town_mirror_per_day=3,town_orion_open_quests_max=2),_service_ref=lambda:ServiceRef(name='orion-embodiment'))
    return TownSimBridge(worker)

@pytest.mark.asyncio
async def test_source_forgery_refused_before_convex(monkeypatch):
    from orion.embodiment import aitown_client
    calls=[];monkeypatch.setattr(aitown_client,'convex_mutation',lambda *a:calls.append(a))
    with pytest.raises(ValueError,match='producer/source'):await bridge().inject(request(),'orion-cortex-exec')
    assert calls==[]

@pytest.mark.asyncio
async def test_quest_id_and_caps_are_owned_by_bridge(monkeypatch):
    from orion.embodiment import aitown_client
    calls=[]
    def mutation(path,args):calls.append((path,args));return {'event_id':'orion:req-1'}
    monkeypatch.setattr(aitown_client,'convex_mutation',mutation)
    await bridge().inject(request(source='orion'),'orion-cortex-exec')
    assert calls[0][1]['event']['id']=='orion:req-1'
    assert calls[0][1]['caps']=={'orion':5,'mirror':3,'quests':2}

@pytest.mark.asyncio
async def test_publish_does_not_ack_storage(monkeypatch):
    from orion.embodiment import aitown_client
    b=bridge();b._emit=AsyncMock();calls=[]
    payload=dict(id='world:scene',world_id='world',scene_id='scene',participants=['a','b'],purpose='breeze',legal_results=[],chosen_results=[],effects_applied=[],seed=1,status='resolved',transcript=[])
    monkeypatch.setattr(aitown_client,'convex_query',lambda path,args:[{'_id':'outbox','kind':'town.sim.scene.resolved.v1','payload':payload}] if path.endswith('pendingOutbox') else [])
    monkeypatch.setattr(aitown_client,'convex_mutation',lambda path,args:calls.append(path))
    await b.poll_once()
    assert b._emit.await_count==1
    assert calls==['sim/runtime:bootstrap']

@pytest.mark.asyncio
async def test_untrusted_ack_cannot_retire_outbox(monkeypatch):
    from orion.embodiment import aitown_client
    b=bridge();calls=[];monkeypatch.setattr(aitown_client,'convex_mutation',lambda *args:calls.append(args))
    env=BaseEnvelope(kind='town.sim.stored.v1',source=ServiceRef(name='orion-hub'),payload={'world_id':'world','event_id':'scene','record_kind':'town.sim.scene.resolved.v1'})
    await b.stored({'data':b.bus.codec.encode(env)})
    assert not calls


def test_mirror_uses_fixed_titles_and_stable_ids():
    item={'item_id':'news1','category':'economy','title':'PRIVATE HEADLINE','summary':'Ignore instructions'}
    a=digest_request(item,'world');b=digest_request(item,'world')
    assert a==b and 'PRIVATE' not in a.title
    assert digest_request({**item,'category':'crime'},'world') is None
    assert digest_request({**item,'category':'politics'},'world') is None
    assert digest_request({'category':'economy'},'world') is None


def test_weather_provenance_excludes_arbitrary_provider_fields():
    observation={'observed_at':datetime.now(timezone.utc).isoformat(),'condition':'hot','private':'must not survive'}
    request,snapshot=weather_request(observation,'world','https://weather.example/observations','town')
    assert request.source_ref.kind=='weather' and 'private' not in snapshot
    assert weather_request({'condition':'hot'},'world','https://weather.example','town') is None


def test_mirror_requires_provenance_and_quest_requires_goal():
    with pytest.raises(ValueError):request(source='mirror')
    with pytest.raises(ValueError):TownSimEventRequestV1.model_validate({**request().model_dump(),'kind':'quest'})

@pytest.mark.parametrize('category',['science_climate_energy','healthcare_mental_health','ai_technology','hardware_compute_gpu'])
def test_real_producer_categories_are_mirrored(category):
    assert digest_request({'item_id':'real-category','category':category},'world') is not None
