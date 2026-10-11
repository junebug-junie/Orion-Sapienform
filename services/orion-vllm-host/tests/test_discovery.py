from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.discovery import pool_server_info
from app.settings import Settings
from app import main


def test_read_only_endpoint_uses_live_engine_state_without_dev_mode():
    app = FastAPI()
    app.middleware('http')(pool_server_info)
    client = TestClient(app)
    assert client.get('/orion/server-info').status_code == 503
    app.state.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model='/actual/model', max_model_len=12345),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=1, data_parallel_size=1),
    )
    result = client.get('/orion/server-info')
    assert result.status_code == 200
    cfg = result.json()['vllm_config']
    assert cfg['model_config'] == {'model':'/actual/model', 'max_model_len':12345}
    assert cfg['scheduler_config']['max_num_seqs'] == 2
    assert client.post('/orion/server-info').status_code == 405
    assert client.get('/server_info').status_code == 404
    assert client.post('/sleep').status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize('values', [dict(LLM_ROLE='agent-deep'), dict(LLM_ANNOUNCE_PORT=8021),
                                    dict(LLM_ROLE='agent-deep', LLM_ANNOUNCE_PORT=8021)])
async def test_incomplete_announcement_refused_before_startup(values, monkeypatch):
    cfg = Settings(_env_file=None, VLLM_PROFILE_NAME=None, **values)
    monkeypatch.setattr(main, 'settings', cfg)
    start = AsyncMock()
    monkeypatch.setattr(main, 'build_heartbeat_chassis', start)
    with pytest.raises(ValueError, match='Pool announcement requires'):
        await main._main_async()
    start.assert_not_called()


@pytest.mark.asyncio
async def test_announcement_uses_existing_bus_contract_and_closes_on_cancel(monkeypatch):
    import asyncio
    from orion.schemas.gpu_pool import LlmWorkerAnnounceV1
    cfg = Settings(_env_file=None, LLM_ROLE='agent-deep', LLM_ANNOUNCE_PORT=8021,
                   VLLM_PROFILE_NAME='glm-test', NODE_NAME='hecate')
    monkeypatch.setattr(main, 'settings', cfg)
    monkeypatch.setattr(Settings, 'resolve_model_and_gpu', lambda self: ('/model', {'cuda_visible_devices':'0,1,2,3'}))
    fake = AsyncMock()
    # One failed bus write must not kill subsequent announcements.
    fake.publish.side_effect = [ConnectionError('bus interrupted'), None]
    monkeypatch.setattr('orion.core.bus.async_service.OrionBusAsync', lambda *a, **kw: fake)
    ticks = 0
    async def tick(_):
        nonlocal ticks
        ticks += 1
        if ticks == 2:
            raise asyncio.CancelledError()
    monkeypatch.setattr(main.asyncio, 'sleep', tick)
    with pytest.raises(asyncio.CancelledError):
        await main.announce_loop()
    assert fake.publish.await_count == 2
    channel, envelope = fake.publish.call_args.args
    assert channel == 'orion:llm:worker:announce'
    payload = LlmWorkerAnnounceV1.model_validate(envelope.payload)
    assert (payload.host, payload.role, payload.port, payload.cuda_visible_devices) == (
        'hecate', 'agent-deep', 8021, '0,1,2,3')
    fake.close.assert_awaited_once()
