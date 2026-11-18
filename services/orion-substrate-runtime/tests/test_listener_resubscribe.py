"""Regression: one Redis error must not end a substrate-runtime listener for good.

Live 2026-10-20: startup blocked the event loop ~3 min, every subscription timed
out, and the finalize_appraisal listener exited while the process stayed up. Every
chat turn then shipped "Response delivered without reflection (substrate appraisal
unavailable (RPC timeout))" until the code changed.
"""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

from app import finalize_appraisal_listener as fal


def test_finalize_appraisal_listener_resubscribes_after_subscription_drop(monkeypatch):
    stop = asyncio.Event()
    handled: list[dict] = []

    class _PubSub:
        def __init__(self, script):
            self.script = list(script)

        async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
            if not self.script:
                stop.set()
                return None
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    sessions = [
        _PubSub([RuntimeError("Timeout reading from socket")]),
        _PubSub([{"type": "message", "data": b"x"}]),
    ]
    calls = {"n": 0}

    class _Bus:
        @contextlib.asynccontextmanager
        async def subscribe(self, channel):
            calls["n"] += 1
            yield sessions.pop(0)

    async def fake_handle(bus, msg, settings):
        handled.append(msg)

    monkeypatch.setattr(fal, "_handle_bus_message", fake_handle)
    settings = SimpleNamespace(
        orion_bus_enabled=True,
        channel_finalize_appraisal_request="orion:substrate:finalize_appraisal:request",
    )

    asyncio.run(
        asyncio.wait_for(
            fal.run_finalize_appraisal_listener(_Bus(), stop, settings=settings), timeout=10
        )
    )
    assert calls["n"] == 2
    assert handled == [{"type": "message", "data": b"x"}]
