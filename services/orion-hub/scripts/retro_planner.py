"""Optional asynchronous tactical choice using Orion's existing LLM gateway.

Only bounded game facts are sent. No chat/recall, frame images, tools or memory writes.
Never runs in the emulator or input relay task.
"""
import asyncio
import json
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    goal: Literal['wait', 'walk_left', 'walk_right', 'jump_left', 'jump_right']
    rationale: str = Field(min_length=1, max_length=512)


async def choose(config, attempt):
    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
    from orion.core.llm_json import parse_json_object
    bus = OrionBusAsync(str(config.ORION_BUS_URL))
    async def request():
        await bus.connect()
        correlation = str(uuid4())
        reply = f'orion:exec:result:{config.LLM_GATEWAY_SERVICE_NAME}:{correlation}'
        prompt = (
            'Choose one supported five-second game motor goal from this observed attempt. '
            'This is Nova the Squirrel only when observation.status is observed. '
            'If unknown, choose wait; do not invent game state. '
            'No horizontal progress after walking can justify testing a jump in that direction. '
            'Position change is not proof of success or reaching an exit. '
            'The motor pulses jump while moving for jump_left/jump_right. '
            'Reply only JSON with goal and a brief evidence-grounded rationale. Facts: '
            + json.dumps(attempt, separators=(',', ':')))
        payload = ChatRequestPayload(messages=[LLMMessage(role='user', content=prompt)],
            route='metacog_background', options={'max_tokens': 256, 'purpose': 'retro_goal',
                'skip_spark_candidate_publish': True,
                'chat_template_kwargs': {'enable_thinking': False}})
        envelope = BaseEnvelope(kind='llm.chat.request',
            source=ServiceRef(name='orion-hub-retro', version='1', node=config.NODE_NAME),
            correlation_id=correlation, reply_to=reply, payload=payload.model_dump(mode='json'))
        msg = await bus.rpc_request(config.CHANNEL_LLM_INTAKE, envelope,
                                    reply_channel=reply, timeout_sec=20)
        decoded = bus.codec.decode(msg.get('data'))
        if not decoded.ok: raise ValueError('planner_invalid_envelope')
        data = decoded.envelope.payload
        if not (data.get('content') or data.get('text')):
            raise ValueError('planner_empty_result:'+str(data.get('raw', {}).get('error', 'no_content'))[:200])
        choice = Choice.model_validate(parse_json_object(data.get('content') or data.get('text') or ''))
        if attempt['observation'].get('status') != 'observed' and choice.goal != 'wait':
            raise ValueError('planner_unobserved_action')
        return choice.model_dump() | {'planner_id': 'llm-gateway:metacog_background:'+correlation}
    try:
        return await asyncio.wait_for(request(), 22)
    finally:
        await bus.close()
