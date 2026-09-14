import asyncio
import threading
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import pytest
from test_generate import main_mod as m, FakePipe

@pytest.fixture(autouse=True)
def state(monkeypatch):
    monkeypatch.setattr(m,"_draining",False)
    monkeypatch.setattr(m,"_generation_future",None)
    monkeypatch.setattr(m,"_generation_lock",asyncio.Lock())
    monkeypatch.setattr(m,"_pipe",FakePipe())
    monkeypatch.setattr(m.settings,"DIFFUSION_DRAIN_TOKEN","test-token")


def test_authenticated_atomic_drain_and_resume():
    client=TestClient(m.app)
    assert client.post('/v1/lifecycle/drain',json={"draining":True}).status_code == 401
    headers={"Authorization":"Bearer test-token"}
    assert client.post('/v1/lifecycle/drain',json={"draining":True},headers=headers).json()["draining"]
    assert client.get('/ready').status_code == 503
    deferred=client.post('/generate',json={"prompt":"sample"})
    assert deferred.status_code == 503 and deferred.json()["reason"] == "controller_displacement"
    assert not client.post('/v1/lifecycle/drain',json={"draining":False},headers=headers).json()["draining"]
    assert client.post('/generate',json={"prompt":"sample","width":8,"height":8}).status_code == 200


def test_cancelled_waiter_retains_real_gpu_occupancy(monkeypatch):
    entered=threading.Event();release=threading.Event()
    def work(_):
        entered.set();release.wait(5);return b'png'
    monkeypatch.setattr(m,"_run_generation",work)
    monkeypatch.setattr(m,"_publish_power_intent",AsyncMock())
    async def scenario():
        task=asyncio.create_task(m.generate(m.GenerateRequest(prompt="sample")))
        await asyncio.to_thread(entered.wait,2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        await m.drain(m.DrainRequest(draining=True),"Bearer test-token")
        assert (await m.lifecycle_status())["in_flight"] is True
        assert (await m.generate(m.GenerateRequest(prompt="new"))).status_code == 503
        release.set()
        await m._generation_future
        assert (await m.lifecycle_status())["in_flight"] is False
        assert (await m.lifecycle_status())["draining"] is True
    try: asyncio.run(scenario())
    finally: release.set()
