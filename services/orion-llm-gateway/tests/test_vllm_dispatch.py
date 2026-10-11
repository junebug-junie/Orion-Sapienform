from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main, pool_placement
from app.models import ChatBody
from app.settings import settings


@pytest.fixture
def vllm_pool(fake_pool, monkeypatch):
    cfg = pool_placement.pool_config().model_copy(deep=True)
    cfg.roles['agent'].backend = 'vllm'
    monkeypatch.setattr(pool_placement, 'pool_config', lambda: cfg)
    fake_pool.choose = lambda kw: 'agent'
    old = fake_pool.grant
    fake_pool.grant = lambda role, n: old(role, n).model_copy(update={'model_file': 'glm-5.3-flash'})
    monkeypatch.setattr(settings, 'llm_lane_routing_enabled', False)
    return fake_pool


@pytest.mark.asyncio
async def test_bus_uses_granted_backend_and_exact_alias_even_on_spill(vllm_pool, monkeypatch):
    seen = []
    def run(body, plan):
        seen.append(plan.route_target)
        return {'text': '391', 'raw': {'model': plan.route_target.model}}
    monkeypatch.setattr(main, 'run_llm_chat', run)
    result = await main._dispatch_chat(ChatBody(route='quick', messages=[{'role':'user','content':'17*23'}]),
                                       correlation_id='vllm-test')
    assert result['text'] == '391'
    assert seen[0].backend == 'vllm' and seen[0].model == 'glm-5.3-flash'
    assert vllm_pool.releases == ['ok']


def test_openai_forwards_discovered_model_and_tools(vllm_pool):
    seen = []
    async def post(self, url, **kwargs):
        seen.append(kwargs['json'])
        return httpx.Response(200, json={'model':'glm-5.3-flash','choices':[{'message':{'content':'391'}}]})
    with patch('httpx.AsyncClient.post', post):
        response = TestClient(main.app).post('/v1/chat/completions', json={
            'model':'agent', 'messages':[{'role':'user','content':'17*23'}],
            'tools':[{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}],
            'chat_template_kwargs':{'enable_thinking':False}})
    assert response.status_code == 200
    assert seen[0]['model'] == 'glm-5.3-flash'
    assert seen[0]['tools'][0]['function']['name'] == 'lookup'
    assert seen[0]['chat_template_kwargs'] == {'enable_thinking':False}
    assert vllm_pool.active == 0


def test_anthropic_refuses_vllm_without_network_or_leaked_lease(vllm_pool):
    with patch('httpx.AsyncClient.post', side_effect=AssertionError('must not call vLLM /v1/messages')):
        response = TestClient(main.app).post('/v1/messages', json={
            'model':'agent','max_tokens':32,'messages':[{'role':'user','content':'hi'}]})
    assert response.status_code == 400
    assert response.json()['error']['type'] == 'unsupported_backend_protocol'
    assert vllm_pool.active == 0 and vllm_pool.releases == ['ok']


@pytest.fixture
def local_vllm(vllm_pool):
    """Real loopback HTTP, no GPU or live Orion bus; exercises gateway wire handling."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            valid = self.path == '/v1/chat/completions' and body.get('model') == 'glm-5.3-flash'
            self.send_response(200 if valid else 400)
            stream = body.get('stream', False)
            self.send_header('Content-Type', 'text/event-stream' if stream else 'application/json')
            self.end_headers()
            if not valid:
                self.wfile.write(b'{"error":{"message":"wrong model or path"}}')
            elif stream:
                self.wfile.write(b'data: {"model":"glm-5.3-flash","choices":[{"delta":{"content":"391"}}]}\n\ndata: [DONE]\n\n')
            else:
                self.wfile.write(json.dumps({'model':'glm-5.3-flash','choices':[{'message':{
                    'role':'assistant', 'content':'391', 'reasoning_content':'Multiply 17 by 23.'},
                    'finish_reason':'stop'}], 'usage':{'prompt_tokens':12,'completion_tokens':2,'total_tokens':14}}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    vllm_pool.urls['agent'] = f'http://127.0.0.1:{server.server_port}'
    try:
        yield seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_bus_completion_over_real_http_preserves_visible_answer_and_model(local_vllm, vllm_pool):
    result = await main._dispatch_chat(ChatBody(route='agent', model='wrong-request-label',
        messages=[{'role':'user','content':'17*23'}],
        options={'max_tokens':32, 'chat_template_kwargs':{'enable_thinking':False}}), correlation_id='loopback-bus')
    assert result['text'] == '391'
    assert result['model'] == 'glm-5.3-flash'
    assert local_vllm[0]['model'] == 'glm-5.3-flash'
    assert local_vllm[0]['chat_template_kwargs'] == {'enable_thinking':False}
    assert vllm_pool.active == 0 and vllm_pool.releases == ['ok']


def test_streamed_completion_over_real_http_releases_lease(local_vllm, vllm_pool):
    with TestClient(main.app).stream('POST', '/v1/chat/completions', json={
        'model':'agent','stream':True,'messages':[{'role':'user','content':'17*23'}]}) as response:
        body = response.read().decode()
        assert response.status_code == 200 and '391' in body and '[DONE]' in body
    assert local_vllm[0]['model'] == 'glm-5.3-flash'
    assert vllm_pool.active == 0 and vllm_pool.releases == ['ok']
