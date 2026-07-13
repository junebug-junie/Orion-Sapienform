from __future__ import annotations

import builtins
import importlib
import logging
import sys

import pytest

from orion.core.bus.resilience import publish_with_reconnect


class _FlakyBus:
    def __init__(self) -> None:
        self.publish_calls = 0
        self.reconnect_calls = 0
        self.last_channel = ""

    async def publish(self, channel: str, msg: object) -> None:
        self.publish_calls += 1
        if self.publish_calls == 1:
            raise TimeoutError("Timeout connecting to server")
        self.last_channel = channel

    async def reconnect(self) -> None:
        self.reconnect_calls += 1


@pytest.mark.asyncio
async def test_publish_with_reconnect_retries_after_transport_error() -> None:
    bus = _FlakyBus()
    await publish_with_reconnect(bus, "orion:pad:stats", {"ok": True}, log_label="test")
    assert bus.reconnect_calls == 1
    assert bus.publish_calls == 2
    assert bus.last_channel == "orion:pad:stats"


def test_resilience_importable_without_loguru(monkeypatch: pytest.MonkeyPatch) -> None:
    """orion.core.bus.resilience must not crash-on-import when loguru is missing.

    Mirrors the fallback already established in bus_service_chassis.py: if
    `import loguru` fails, fall back to the stdlib logging module so the
    caller's try/except around the import (e.g. worker.py's
    _publish_brain_frame) never has to fire in the first place.
    """
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "loguru" or name.startswith("loguru."):
            raise ImportError("No module named 'loguru'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    import orion.core.bus.resilience as resilience_module

    # Only drop the cached loguru module so the fallback's `import loguru`
    # actually executes (and hits our blocked import) instead of resolving
    # from cache. Deliberately keep `orion.core.bus.resilience` itself in
    # sys.modules and reload it in place, so the module object identity
    # never changes and this test's cleanup reload can't desync from it.
    monkeypatch.delitem(sys.modules, "loguru", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    importlib.reload(resilience_module)

    try:
        assert resilience_module.logger is not None
        assert isinstance(resilience_module.logger, logging.Logger)
        assert hasattr(resilience_module, "publish_with_reconnect")
    finally:
        # Restore the real import and reload with loguru available again so
        # later tests in the same process see the normal loguru-backed logger.
        monkeypatch.undo()
        importlib.reload(resilience_module)
