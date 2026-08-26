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


def test_startup_raises_when_gpu_expected_but_cuda_unavailable(monkeypatch):
    """The actual incident this closes: a container that boots with CUDA
    already broken must fail LOUD at startup, not silently serve requests
    until the first real TTS call crashes."""
    module, _ = _load_main_module(monkeypatch, cuda_available=False)
    module.settings.tts_use_gpu = True
    bus_cls, _ = _fake_bus_cls()
    monkeypatch.setattr(module, "OrionBusAsync", bus_cls)
    monkeypatch.setattr(module, "listener_worker", AsyncMock())
    monkeypatch.setattr(module, "stt_listener_worker", AsyncMock())

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        asyncio.run(module.startup())


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
