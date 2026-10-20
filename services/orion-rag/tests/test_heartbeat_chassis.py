"""orion-rag publishes the standard SystemHealthV1 heartbeat via HeartbeatOnly.

Before this patch a private thread ran its own asyncio loop publishing every 30s
with the made-up node name "rag-node". The heartbeat now comes from
orion.core.bus.bus_service_chassis.HeartbeatOnly, started/stopped by the app's own
startup/shutdown hooks. Heavy deps (chromadb, sentence-transformers) are stubbed out
the same way test_listener_resubscribe.py does.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

for _key, _value in {
    "SERVICE_NAME": "orion-rag",
    "SERVICE_VERSION": "test",
    "PORT": "0",
    "ORION_BUS_ENABLED": "true",
    "ORION_BUS_URL": "redis://unused:6379/0",
    "SUBSCRIBE_CHANNEL_RAG_REQUEST": "orion:rag:request",
    "PUBLISH_CHANNEL_BRAIN_INTAKE": "orion:brain:intake",
    "VECTOR_DB_URL": "http://unused:8000",
    "VECTOR_DB_COLLECTION": "test",
    "EMBEDDING_MODEL": "test",
}.items():
    os.environ.setdefault(_key, _value)

_processor = types.ModuleType("app.processor")
_processor.process_rag_request = lambda data: None
sys.modules.setdefault("app.processor", _processor)
_vector_store_mod = types.ModuleType("app.vector_store")
_vector_store_mod.vector_store = MagicMock()
sys.modules.setdefault("app.vector_store", _vector_store_mod)

from orion.core.bus.bus_service_chassis import HeartbeatOnly  # noqa: E402

from app import main  # noqa: E402
from app.settings import settings  # noqa: E402


def test_build_heartbeat_chassis_uses_service_settings() -> None:
    chassis = main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.SERVICE_NAME
    assert chassis.cfg.service_version == settings.SERVICE_VERSION
    assert chassis.cfg.node_name == settings.NODE_NAME
    assert chassis.cfg.node_name != "rag-node"
    assert chassis.cfg.bus_url == settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


def test_old_heartbeat_thread_is_gone() -> None:
    assert not hasattr(main, "heartbeat_worker_thread")


def test_startup_starts_and_shutdown_stops_heartbeat(monkeypatch) -> None:
    fake = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: fake)
    monkeypatch.setattr(main, "listener_worker", lambda: None)  # no real listener thread
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", True)

    async def _run():
        await main.app.router.startup()
        assert main.heartbeat_chassis is fake
        fake.start_background.assert_awaited_once()
        await main.app.router.shutdown()

    asyncio.run(_run())
    fake.stop.assert_awaited_once()
    assert main.heartbeat_chassis is None


def test_heartbeat_start_failure_is_not_fatal(monkeypatch) -> None:
    fake = AsyncMock(spec=HeartbeatOnly)
    fake.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: fake)
    monkeypatch.setattr(main, "listener_worker", lambda: None)
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", True)

    async def _run():
        await main.app.router.startup()  # must not raise
        assert main.heartbeat_chassis is None
        await main.app.router.shutdown()

    asyncio.run(_run())
    fake.stop.assert_not_awaited()


def test_bus_disabled_starts_no_heartbeat(monkeypatch) -> None:
    built = []
    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: built.append(1))
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", False)

    async def _run():
        await main.app.router.startup()
        await main.app.router.shutdown()

    asyncio.run(_run())
    assert built == []
    assert main.heartbeat_chassis is None
