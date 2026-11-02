"""orion-thought's hand-written listener loops must leave a silently dead pub/sub line.

Their subscriber-count check ran only after a read overran its timeout. A dead line
answers "nothing yet" on time, so that check never ran. The shared PubSubWatchdog PINGs
after quiet and forces a reconnect when no pong comes back.
"""
from __future__ import annotations

import asyncio
import contextlib

import pytest

from orion.core.bus import async_service


class _Line:
    def __init__(self, *, alive: bool) -> None:
        self.alive = alive
        self.queue: list[dict] = []
        self.pings = 0

    async def get_message(self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0):
        if self.queue:
            return self.queue.pop(0)
        await asyncio.sleep(min(timeout or 0, 0.01))
        return None

    async def ping(self) -> None:
        self.pings += 1
        if self.alive:
            self.queue.append({"type": "pong", "pattern": None, "channel": None, "data": b""})


def _fake_bus_class(lines: list[_Line], stop: asyncio.Event):
    class _Bus:
        def __init__(self, url: str = "", **_kw) -> None:
            pass

        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

        def subscribe(self, *channels: str, patterns: bool = False):
            @contextlib.asynccontextmanager
            async def _cm():
                line = _Line(alive=bool(lines))  # first line is dead, later ones live
                lines.append(line)
                if len(lines) == 2:
                    stop.set()
                yield line

            return _cm()

    return _Bus


@pytest.fixture
def fast_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(async_service, "PUBSUB_IDLE_PING_SEC", 0.05)
    monkeypatch.setattr(async_service, "PUBSUB_PONG_DEADLINE_SEC", 0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_name,worker_name",
    [("app.bus_listener", "run_bus_worker"), ("app.reasoning_activity", "run_reasoning_worker")],
)
async def test_worker_reconnects_off_a_silent_line(
    monkeypatch: pytest.MonkeyPatch, fast_watchdog, module_name: str, worker_name: str
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.settings, "orion_bus_enabled", True, raising=False)
    stop = asyncio.Event()
    lines: list[_Line] = []
    monkeypatch.setattr(module, "OrionBusAsync", _fake_bus_class(lines, stop))

    await asyncio.wait_for(getattr(module, worker_name)(stop_event=stop), timeout=10)

    assert len(lines) == 2, "worker stayed on the dead line"
    assert lines[0].pings == 1
