"""Shared bus client; callers cannot choose their trust class."""
from uuid import uuid4
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.town_sim import TownSimEventRequestV1, TownSimEventResultV1

async def request_town_event(bus, payload:dict, *, producer:str, node:str='unknown') -> dict:
    source={'orion-hub':'juniper','orion-cortex-exec':'orion'}[producer]
    correlation=str(uuid4())
    request=TownSimEventRequestV1.model_validate({**payload,'source':source,'requested_by':producer,'correlation_id':payload.get('correlation_id') or correlation})
    reply=f'orion:town:sim:reply:{correlation}'
    env=BaseEnvelope(kind='town.sim.event.request.v1',source=ServiceRef(name=producer,node=node),correlation_id=correlation,reply_to=reply,payload=request.model_dump(mode='json',exclude_none=True))
    raw=await bus.rpc_request('orion:town:sim:event:request',env,reply_channel=reply,timeout_sec=10)
    decoded=bus.codec.decode(raw.get('data'))
    if not decoded.ok or decoded.envelope.kind!='town.sim.event.result.v1':raise ValueError('invalid town response')
    result=TownSimEventResultV1.model_validate(decoded.envelope.payload)
    if result.correlation_id!=request.correlation_id or result.world_id!=request.world_id:raise ValueError('mismatched town response')
    return result.model_dump(mode='json')
