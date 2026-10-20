"""whisper-tts publishes the standard SystemHealthV1 heartbeat via HeartbeatOnly.

Before this patch a hand-rolled 30s loop shared the TTS/STT listeners' bus and
reported the made-up node name "whisper-node" (live in equilibrium's state hash as
`whisper-tts@whisper-node`, 2026-10-20). The heartbeat now comes from
orion.core.bus.bus_service_chassis.HeartbeatOnly on its own bus connection, with the
real NODE_NAME, started in startup() and stopped in shutdown(). torch/whisper and the
listener bus are stubbed the same way test_main_startup_cuda_gating.py does.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus.bus_service_chassis import HeartbeatOnly

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _load_main(monkeypatch: pytest.MonkeyPatch):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisper", MagicMock())
    for mod_name in ("app.main", "app.tts_worker", "app.stt_worker", "app"):
        sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location("app.main", SERVICE_ROOT / "app" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["app.main"] = module
    spec.loader.exec_module(module)
    module.settings.tts_use_gpu = False  # no CUDA guard / watchdog in this test

    bus = MagicMock()
    bus.connect = AsyncMock()
    bus.close = AsyncMock()
    monkeypatch.setattr(module, "OrionBusAsync", MagicMock(return_value=bus))

    async def _hang(*a, **k):
        await asyncio.sleep(3600)

    monkeypatch.setattr(module, "listener_worker", _hang)
    monkeypatch.setattr(module, "stt_listener_worker", _hang)
    return module


@pytest.fixture(autouse=True)
def _cleanup_sys_modules():
    yield
    sys.modules.pop("app.main", None)


def test_build_heartbeat_chassis_uses_service_settings(monkeypatch) -> None:
    module = _load_main(monkeypatch)
    chassis = module.build_heartbeat_chassis()
    s = module.settings
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == s.service_name == "whisper-tts"
    assert chassis.cfg.service_version == s.service_version
    assert chassis.cfg.node_name == s.node_name
    assert chassis.cfg.node_name != "whisper-node"
    assert chassis.cfg.bus_url == s.orion_bus_url
    assert chassis.cfg.bus_enabled == s.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == s.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"
    # /health and the bus heartbeat report the same process-run id.
    assert chassis.boot_id == module.BOOT_ID


def test_old_hand_rolled_heartbeat_loop_is_gone(monkeypatch) -> None:
    module = _load_main(monkeypatch)
    assert not hasattr(module, "heartbeat_loop")
    assert not hasattr(module, "heartbeat_task")


def test_startup_starts_and_shutdown_stops_heartbeat(monkeypatch) -> None:
    module = _load_main(monkeypatch)
    fake = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(module, "build_heartbeat_chassis", lambda: fake)

    async def _run():
        await module.startup()
        assert module.heartbeat_chassis is fake
        fake.start_background.assert_awaited_once()
        await module.shutdown()

    asyncio.run(_run())
    fake.stop.assert_awaited_once()
    assert module.heartbeat_chassis is None


def test_heartbeat_start_failure_is_not_fatal(monkeypatch) -> None:
    module = _load_main(monkeypatch)
    fake = AsyncMock(spec=HeartbeatOnly)
    fake.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(module, "build_heartbeat_chassis", lambda: fake)

    async def _run():
        await module.startup()  # must not raise
        assert module.heartbeat_chassis is None
        assert module.listener_task is not None and not module.listener_task.done()
        await module.shutdown()

    asyncio.run(_run())
    fake.stop.assert_not_awaited()
