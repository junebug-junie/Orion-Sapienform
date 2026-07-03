from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_social_stored_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _source() -> ServiceRef:
    return ServiceRef(name="test-hub", version="0.0.1", node="local")


class _DisconnectedUntilReconnectBus:
    """Simulates concurrent-handler task context where bus is not connected."""

    def __init__(self) -> None:
        self.connected = False
        self.publish_calls = 0
        self.reconnect_calls = 0
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, msg: BaseEnvelope) -> None:
        self.publish_calls += 1
        if not self.connected:
            raise RuntimeError("OrionBusAsync not connected. Call await connect().")
        self.published.append((channel, msg))

    async def reconnect(self) -> None:
        self.reconnect_calls += 1
        self.connected = True


@pytest.mark.asyncio
async def test_social_turn_stored_reconnects_when_bus_disconnected(monkeypatch):
    corr = str(uuid4())
    bus = _DisconnectedUntilReconnectBus()

    async def _fake_write(*args, **kwargs):
        return True

    monkeypatch.setattr(worker, "_write", _fake_write)
    monkeypatch.setattr(worker.settings, "sql_writer_emit_social_turn_stored", True)
    monkeypatch.setattr(
        worker.settings,
        "sql_writer_social_turn_stored_channel",
        "orion:chat:social:stored",
    )

    env = BaseEnvelope(
        kind="social.turn.v1",
        correlation_id=corr,
        source=_source(),
        payload={
            "correlation_id": corr,
            "prompt": "vision learning",
            "response": "I'd like to explore that with you",
            "session_id": "sess-social",
        },
    )

    await worker.handle_envelope(env, bus=bus)

    assert bus.reconnect_calls == 1
    assert bus.publish_calls == 2
    assert len(bus.published) == 1
    channel, out_env = bus.published[0]
    assert channel == "orion:chat:social:stored"
    assert out_env.kind == worker.SOCIAL_TURN_STORED_KIND
    assert str(out_env.correlation_id) == corr
