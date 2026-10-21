"""Public-feed -> closed town templates. No private-context imports or LLM rewrites."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from orion.schemas.town_sim import TownEffect, TownSimEventRequestV1, TownSourceRef

# Deliberately fixed text: news titles and personal terms never enter town speech.
TEMPLATES = {
    'economy': ('late_shipment', 'A parts shipment is running late.', [dict(op='fact_set',target='shipment.cancelled',value=True)]),
    'business': ('price_shift', 'Supplies at the diner are running low.', [dict(op='stock_delta',target='diner.coffee',value=-1)]),
    'science': ('rumor', 'There is a new idea to discuss in the square.', []),
    'space': ('rumor', 'There is news from the sky to discuss in the square.', []),
    'health': ('rumor', 'The town is taking a quiet rest day.', []),
}

def digest_request(item: dict[str, Any], world_id: str) -> TownSimEventRequestV1 | None:
    category=str(item.get('category','')).lower()
    category={'science_climate_energy':'science','healthcare_mental_health':'health','ai_technology':'science','hardware_compute_gpu':'business'}.get(category,category)
    if category not in TEMPLATES or not item.get('item_id'):
        return None
    kind,title,effects=TEMPLATES[category]
    item_id=str(item['item_id'])
    key=hashlib.sha256(item_id.encode()).hexdigest()
    return TownSimEventRequestV1(correlation_id=key,world_id=world_id,source='mirror',kind=kind,title=title,
        effects=[TownEffect(**e) for e in effects],source_ref=TownSourceRef(kind='digest',id=item_id),requested_by='town_mirror')

def weather_request(observation:dict[str,Any],world_id:str,provider:str,region:str):
    """Provider adapter input: public observation, not WeatherCurrent personal context.

    Accepts a documented compact JSON observation contract. Operators must point
    the URL at a public source that serves this contract (see service README).
    No event is produced for missing/future/stale timestamps or ordinary weather.
    """
    try:
        observed=datetime.fromisoformat(str(observation['observed_at']).replace('Z','+00:00'))
        if observed.tzinfo is None: return None
        age=(datetime.now(timezone.utc)-observed).total_seconds()
        if age<0 or age>21600:return None
        condition=observation['condition']
        if condition not in {'hot','cold','storm','smoke'}:return None
    except (KeyError,TypeError,ValueError):return None
    payload={'provider':provider,'region':region,'observed_at':observed.isoformat(),'condition':condition}
    sid=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    request=TownSimEventRequestV1(correlation_id=sid,world_id=world_id,source='mirror',kind='weather_spell',title=f'The town has {condition} weather.',effects=[TownEffect(op='place_flag',target=f'square|{condition}',value=True)],source_ref=TownSourceRef(kind='weather',id=sid),requested_by='town_mirror')
    return request,payload
