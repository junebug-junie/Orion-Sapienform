#!/usr/bin/env python3
"""CPU-only gate inside the pinned 1Cat image: real protocol classes and template parameters."""
import asyncio
import json
from types import SimpleNamespace

from app.deep_budget import check_request


async def main():
    captured = []
    count = 229376

    async def tokenize(req, raw):
        captured.append(req)
        return SimpleNamespace(count=count)

    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=262144))
    async def call(path, body):
        async def read():
            return body
        req = SimpleNamespace(url=SimpleNamespace(path=path), json=read,
            app=SimpleNamespace(state=SimpleNamespace(openai_serving_tokenization=
                                                      SimpleNamespace(create_tokenize=tokenize))))
        return await check_request(req, cfg)

    body = {'model': 'deep.thinker.v1', 'messages': [{'role': 'user', 'content': 'test'}],
            'reasoning_effort': 'none', 'documents': [{'title': 'record', 'text': 'evidence'}],
            'tools': [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]}
    assert await call('/v1/chat/completions', body) is None
    params = captured[-1].build_chat_params(None, 'string').chat_template_kwargs
    assert params['enable_thinking'] is False and params['documents'] == body['documents']
    assert captured[-1].tools[0].function.name == 'lookup'
    count += 1
    assert (await call('/v1/chat/completions', body)).status_code == 400
    count = 245760
    anthropic = {'model': 'deep.thinker.v1', 'max_tokens': 16384,
        'system': 'retain evidence',
        'tools': [{'name': 'lookup', 'description': 'read', 'input_schema': {'type': 'object'}}],
        'messages': [
            {'role': 'user', 'content': 'read'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'lookup', 'input': {}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'evidence'}]}]}
    assert await call('/v1/messages', anthropic) is None
    messages = captured[-1].model_dump()['messages']
    assert messages[0]['role'] == 'system' and messages[-1]['role'] == 'tool'
    assert messages[-1]['content'] == 'evidence'
    count += 1
    assert (await call('/v1/messages', anthropic)).status_code == 400
    assert (await call('/v1/responses', {})).status_code == 400
    assert (await call('/v1/completions', {})).status_code == 400
    print(json.dumps({'real_protocol_boundary_checks': 'passed', 'gpu_inference': 'UNVERIFIED'}))


if __name__ == '__main__':
    asyncio.run(main())
