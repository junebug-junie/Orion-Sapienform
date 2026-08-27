"""2026-08-26: `startup()` actually calls the boot-time CUDA guard and starts
the watchdog task, gated correctly on `tts_use_gpu`.

`_require_cuda_or_die()` existed in `main.py` for an unknown amount of time
with a comment reading "call this during startup before spinning workers"
and was never actually called anywhere -- confirmed dead code, found while
investigating the 2026-08-26 TTS outage. A test asserting only that the
function itself raises correctly (the arithmetic) would not have caught
that it was never wired in (the lifecycle) -- see
`feedback_test_the_lifecycle_not_just_the_arithmetic` in project memory.
This file exercises the real `startup()` coroutine, not a reimplementation
of its logic.

Heavy/network dependencies are stubbed rather than skipped: `torch` is not
installed in the general dev venv (this service's own container image
installs a CUDA build), and `OrionBusAsync.connect()` is a real network
call. Both are faked so this test exercises real wiring with no network and
no GPU.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _load_main_module(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool):
    """Stubs torch (controllable cuda.is_available) the same way
    test_stt_engine.py already does for this service, then loads app.main
    fresh via importlib so each test gets an independent module object --
    module-level globals (bus, *_task) must not leak between tests.
    """
    fake_torch = MagicMock()
    fake_torch.version.cuda = "12.6"
    fake_torch.backends.cuda.is_built.return_value = True
    fake_torch.cuda.is_available.return_value = cuda_available
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # stt.py imports `whisper` at module level (tts.py's TTS import is lazy,
    # inside CoquiBackend.__init__, so no stub needed for that one) -- same
    # stub test_stt_engine.py already uses for this exact reason.
    monkeypatch.setitem(sys.modules, "whisper", MagicMock())
    for mod_name in ("app.main", "app.tts_worker", "app.stt_worker", "app"):
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(
        "app.main", SERVICE_ROOT / "app" / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["app.main"] = module
    spec.loader.exec_module(module)
    return module, fake_torch


def _fake_bus_cls():
    """A stand-in OrionBusAsync: connect()/close() are no-ops, nothing
    touches a real Redis."""
    instance = MagicMock()
    instance.connect = AsyncMock()
    instance.close = AsyncMock()
    instance.redis = None
    cls = MagicMock(return_value=instance)
    return cls, instance


@pytest.fixture(autouse=True)
def _cleanup_sys_modules():
    yield
    for mod_name in ("app.main",):
        sys.modules.pop(mod_name, None)


def test_startup_logs_critical_but_does_not_raise_when_cuda_unavailable(monkeypatch, caplog):
    """Review finding, 2026-08-26: an earlier version of this patch let this
    raise and crash the WHOLE process -- bus, listener_task, stt_task, all
    of it -- before any of them started. STT does not need CUDA at all
    (stt.py falls back to CPU); the incident this patch closes is that STT
    SURVIVED a CUDA staleness event that killed TTS. A hard crash-at-boot
    took away exactly that resilience. The boot guard must log loud and let
    startup continue -- enforcement is the watchdog's job alone."""
    import logging

    module, _ = _load_main_module(monkeypatch, cuda_available=False)
    module.settings.tts_use_gpu = True
    module.settings.cuda_watchdog_enabled = True
    bus_cls, bus_instance = _fake_bus_cls()
    monkeypatch.setattr(module, "OrionBusAsync", bus_cls)

    async def _hang(*a, **k):
        await asyncio.sleep(3600)

    monkeypatch.setattr(module, "listener_worker", _hang)
    monkeypatch.setattr(module, "stt_listener_worker", _hang)

    async def _run():
        with caplog.at_level(logging.CRITICAL, logger="orion-whisper-tts"):
            await module.startup()  # must NOT raise
        assert module.listener_task is not None and not module.listener_task.done()
        assert module.stt_task is not None and not module.stt_task.done()
        # The watchdog still starts -- it is the actual enforcement path,
        # and will restart the process once its own failure_threshold is
        # reached (covered by test_cuda_watchdog.py's loop tests).
        assert module.cuda_watchdog_task is not None
        await module.shutdown()

    asyncio.run(_run())
    assert any("cuda_unavailable_at_boot" in r.getMessage() for r in caplog.records)
    bus_instance.connect.assert_awaited_once()


def test_startup_succeeds_and_starts_watchdog_when_cuda_available(monkeypatch):
    module, fake_torch = _load_main_module(monkeypatch, cuda_available=True)
    module.settings.tts_use_gpu = True
    module.settings.cuda_watchdog_enabled = True
    bus_cls, bus_instance = _fake_bus_cls()
    monkeypatch.setattr(module, "OrionBusAsync", bus_cls)

    async def _hang(*a, **k):
        await asyncio.sleep(3600)

    monkeypatch.setattr(module, "listener_worker", _hang)
    monkeypatch.setattr(module, "stt_listener_worker", _hang)

    async def _run():
        await module.startup()
        assert module.cuda_watchdog_task is not None
        assert not module.cuda_watchdog_task.done()
        await module.shutdown()

    asyncio.run(_run())
    bus_instance.connect.assert_awaited_once()
    bus_instance.close.assert_awaited_once()


def test_startup_skips_both_gpu_checks_when_gpu_mode_is_off(monkeypatch):
    """A deliberate CPU-mode deployment (tts_use_gpu=False) must not be
    forced to have a GPU by either the boot-time guard or the watchdog --
    both are gated on the SAME flag."""
    module, _ = _load_main_module(monkeypatch, cuda_available=False)
    module.settings.tts_use_gpu = False
    bus_cls, bus_instance = _fake_bus_cls()
    monkeypatch.setattr(module, "OrionBusAsync", bus_cls)

    async def _hang(*a, **k):
        await asyncio.sleep(3600)

    monkeypatch.setattr(module, "listener_worker", _hang)
    monkeypatch.setattr(module, "stt_listener_worker", _hang)

    async def _run():
        await module.startup()  # must not raise
        assert module.cuda_watchdog_task is None
        await module.shutdown()

    asyncio.run(_run())


def test_health_endpoint_reports_cuda_state(monkeypatch):
    """Review finding, 2026-08-26: neither /health nor the heartbeat's bus
    publish reflected CUDA state before this -- an operator polling /health
    during the poll_sec*failure_threshold window before a restart would see
    'status: ok' the whole time. This is the fix, tested directly against
    the real FastAPI route function, not a reimplementation of it."""
    module, _ = _load_main_module(monkeypatch, cuda_available=False)
    module.settings.tts_use_gpu = True
    module.settings.cuda_watchdog_enabled = True

    async def _run():
        resp = await module.health()
        return resp

    resp = asyncio.run(_run())
    import json

    body = json.loads(bytes(resp.body))
    assert body["cuda_available"] is False
    assert body["cuda_watchdog_enabled"] is True


def test_health_endpoint_omits_cuda_state_in_cpu_mode(monkeypatch):
    """A deliberate CPU-mode deployment should not report a misleading
    cuda_available=False as though it were a fault -- there is no GPU
    expected in the first place."""
    module, _ = _load_main_module(monkeypatch, cuda_available=False)
    module.settings.tts_use_gpu = False

    async def _run():
        return await module.health()

    resp = asyncio.run(_run())
    import json

    body = json.loads(bytes(resp.body))
    assert body["cuda_available"] is None
    assert body["cuda_watchdog_enabled"] is False


def test_watchdog_disabled_flag_suppresses_the_task_even_with_gpu_on(monkeypatch):
    """CUDA_WATCHDOG_ENABLED=false must be honored independently of
    tts_use_gpu -- an operator opting out of the watchdog specifically
    (e.g. while debugging) must not be overridden."""
    module, _ = _load_main_module(monkeypatch, cuda_available=True)
    module.settings.tts_use_gpu = True
    module.settings.cuda_watchdog_enabled = False
    bus_cls, bus_instance = _fake_bus_cls()
    monkeypatch.setattr(module, "OrionBusAsync", bus_cls)

    async def _hang(*a, **k):
        await asyncio.sleep(3600)

    monkeypatch.setattr(module, "listener_worker", _hang)
    monkeypatch.setattr(module, "stt_listener_worker", _hang)

    async def _run():
        await module.startup()
        assert module.cuda_watchdog_task is None
        await module.shutdown()

    asyncio.run(_run())
