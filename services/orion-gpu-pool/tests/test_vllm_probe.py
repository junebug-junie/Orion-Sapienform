import httpx
import pytest
from app.main import _probe


@pytest.mark.asyncio
async def test_vllm_probe_reads_engine_facts_never_llamacpp_props():
    seen = []
    def handle(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={'data':[]} if request.url.path == '/v1/models' else {})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        probe = await _probe(client, 'agent-deep', 'http://hecate:8021', 'llm', '/health', 'vllm')
    assert probe.ok and probe.backend == 'vllm'
    assert seen == ['http://hecate:8021/health', 'http://hecate:8021/v1/models',
                    'http://hecate:8021/orion/server-info']


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/health', '/v1/models', '/orion/server-info'])
async def test_failed_vllm_probe_is_down(path):
    def handle(request):
        return httpx.Response(503 if request.url.path == path else 200, json={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        probe = await _probe(client, 'agent-deep', 'http://hecate:8021', 'llm', '/health', 'vllm')
    assert not probe.ok
