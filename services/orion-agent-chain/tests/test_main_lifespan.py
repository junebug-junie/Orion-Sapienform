from __future__ import annotations

import asyncio

from app import main as agent_main


def test_lifespan_keeps_bus_task_alive(monkeypatch):
    states = {"started": False, "cancelled": False}

    async def _fake_bus_worker(stop_event):
        states["started"] = True
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            states["cancelled"] = True
            raise

    async def _fake_heartbeat(_settings):
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(agent_main, "run_bus_worker", _fake_bus_worker)
    monkeypatch.setattr(agent_main, "heartbeat_loop", _fake_heartbeat)

    async def _run():
        async with agent_main.lifespan(agent_main.app):
            await asyncio.sleep(0.05)
            assert hasattr(agent_main.app.state, "bus_task")
            assert not agent_main.app.state.bus_task.done()
            assert states["started"] is True

    asyncio.run(_run())
    assert states["started"] is True
