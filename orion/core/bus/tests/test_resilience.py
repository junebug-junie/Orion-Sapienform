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
    await publish_with_reconnect(bus, "orion:system:health", {"ok": True}, log_label="test")
    assert bus.reconnect_calls == 1
    assert bus.publish_calls == 2
    assert bus.last_channel == "orion:system:health"


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


class _ScriptedPubSub:
    """Yields scripted get_message results: dicts are messages, exceptions are raised."""

    def __init__(self, script: list[object], stop) -> None:
        self._script = list(script)
        self._stop = stop

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if not self._script:
            self._stop.set()
            return None
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _ScriptedBus:
    def __init__(self, sessions: list[object]) -> None:
        self.sessions = list(sessions)
        self.subscribe_calls = 0

    def subscribe(self, channel: str):
        import contextlib

        bus = self

        @contextlib.asynccontextmanager
        async def _cm():
            bus.subscribe_calls += 1
            session = bus.sessions.pop(0)
            if isinstance(session, BaseException):
                raise session
            yield session

        return _cm()


def _msg(n: int) -> dict:
    return {"type": "message", "data": n}


@pytest.mark.asyncio
async def test_run_subscription_resubscribes_after_a_mid_stream_drop() -> None:
    """Live 2026-10-20: a redis TimeoutError from get_message ended the listener for good."""
    import asyncio

    from orion.core.bus.resilience import run_subscription

    stop = asyncio.Event()
    seen: list[int] = []

    class _RedisTimeout(Exception):  # redis.exceptions.TimeoutError is not asyncio's
        pass

    first = _ScriptedPubSub([_msg(1), _RedisTimeout("Timeout reading from socket")], stop)
    second = _ScriptedPubSub([_msg(2)], stop)
    bus = _ScriptedBus([first, ConnectionError("bus down"), second])

    async def handle(msg: dict) -> None:
        seen.append(msg["data"])

    await asyncio.wait_for(
        run_subscription(bus, "orion:test", handle, stop_event=stop, name="t", max_backoff_sec=0.01),
        timeout=10,
    )
    assert seen == [1, 2]
    assert bus.subscribe_calls == 3


@pytest.mark.asyncio
async def test_run_subscription_survives_handler_errors_and_skips_non_messages() -> None:
    import asyncio

    from orion.core.bus.resilience import run_subscription

    stop = asyncio.Event()
    seen: list[int] = []

    def handle(msg: dict) -> None:  # sync handler
        if msg["data"] == 1:
            raise ValueError("bad payload")
        seen.append(msg["data"])

    session = _ScriptedPubSub([_msg(1), None, {"type": "subscribe", "data": 9}, _msg(2)], stop)
    bus = _ScriptedBus([session])
    await asyncio.wait_for(
        run_subscription(bus, "orion:test", handle, stop_event=stop, name="t"), timeout=10
    )
    assert seen == [2]
    assert bus.subscribe_calls == 1


@pytest.mark.asyncio
async def test_run_subscription_reopens_a_session_whose_reads_hang() -> None:
    """A dead connection can make every read overrun the guard; that must not poll forever."""
    import asyncio

    from orion.core.bus.resilience import run_subscription

    stop = asyncio.Event()
    seen: list[int] = []

    class _HangingPubSub:
        async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
            raise asyncio.TimeoutError  # what the outer wait_for raises on an overrun

    bus = _ScriptedBus([_HangingPubSub(), _ScriptedPubSub([_msg(7)], stop)])
    await asyncio.wait_for(
        run_subscription(
            bus, "orion:test", seen.append, stop_event=stop, name="t",
            max_backoff_sec=0.01, stuck_reads=3,
        ),
        timeout=10,
    )
    assert seen == [_msg(7)]
    assert bus.subscribe_calls == 2


@pytest.mark.asyncio
async def test_run_subscription_propagates_cancellation() -> None:
    import asyncio

    from orion.core.bus.resilience import run_subscription

    class _QuietPubSub:
        async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
            await asyncio.sleep(0.01)
            return None

    task = asyncio.create_task(
        run_subscription(_ScriptedBus([_QuietPubSub()]), "orion:test", print, stop_event=None, name="t")
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_subscription_health_reports_down_while_backing_off_and_up_after() -> None:
    """A listener stuck retrying must read as down, not healthy (2026-10-20)."""
    import asyncio

    from orion.core.bus import resilience
    from orion.core.bus.resilience import run_subscription, subscription_health

    resilience._SUBSCRIPTION_STATES.pop("health-t", None)
    stop = asyncio.Event()
    gate = asyncio.Event()
    snapshots: list[dict] = []

    class _GatedPubSub:
        async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
            snapshots.append(subscription_health())
            if not gate.is_set():
                gate.set()
                return _msg(1)
            stop.set()
            return None

    bus = _ScriptedBus([ConnectionError("bus down"), _GatedPubSub()])
    task = asyncio.create_task(
        run_subscription(bus, "orion:test", lambda m: None, stop_event=stop, name="health-t", max_backoff_sec=0.2)
    )
    for _ in range(50):
        row = next((r for r in subscription_health()["listeners"] if r["name"] == "health-t"), None)
        if row and row["failures_total"]:
            break
        await asyncio.sleep(0.01)
    during = subscription_health()
    assert "health-t" in during["down"] and during["ok"] is False
    assert row["last_failure"].startswith("ConnectionError")

    await asyncio.wait_for(task, timeout=10)
    up = next(r for r in snapshots[0]["listeners"] if r["name"] == "health-t")
    assert up["subscribed"] is True and "health-t" not in snapshots[0]["down"]
    done = next(r for r in subscription_health()["listeners"] if r["name"] == "health-t")
    assert done["messages_total"] == 1 and done["running"] is False and done["subscribed"] is False
    assert done["last_message_age_sec"] is not None
    resilience._SUBSCRIPTION_STATES.pop("health-t", None)
