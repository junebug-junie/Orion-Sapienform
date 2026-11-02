"""Hunter must not cancel in-flight concurrent handlers when only the pub/sub line died
(iter_messages raising PubSubSilentError); it still cancels them on stop."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus.async_service import PubSubSilentError
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter


def _hunter(handler) -> Hunter:
    cfg = ChassisConfig(service_name="t", service_version="0", node_name="n",
                        bus_url="redis://localhost:6379/0", bus_enabled=True)
    h = Hunter(cfg, handler=handler, patterns=["orion:x"], concurrent_handlers=True)
    h.bus = MagicMock()
    h.bus.enabled = True
    h.bus.redis = object()
    h.bus.reconnect = AsyncMock()
    env = BaseEnvelope(kind="k", source=ServiceRef(name="s", node="n"), payload={})
    h.bus.codec.decode = MagicMock(return_value=MagicMock(ok=True, envelope=env))
    h._publish_error = AsyncMock()

    @asynccontextmanager
    async def subscribe(*_a, **_k):
        yield MagicMock()

    h.bus.subscribe = subscribe
    return h


@pytest.mark.asyncio
async def test_inflight_handler_survives_dead_line_and_resubscribe() -> None:
    finished: list[str] = []
    release = asyncio.Event()

    async def handler(_env) -> None:
        await release.wait()
        finished.append("done")

    h = _hunter(handler)
    calls = {"n": 0}

    async def iter_messages(_ps):
        calls["n"] += 1
        if calls["n"] == 1:
            yield {"type": "message", "channel": b"orion:x", "data": b"x"}
            await asyncio.sleep(0)
            raise PubSubSilentError("pubsub silent")
        release.set()
        await asyncio.sleep(0.05)
        h._stop.set()
        return
        yield  # pragma: no cover

    h.bus.iter_messages = iter_messages
    await asyncio.wait_for(h._run(), timeout=5)
    assert calls["n"] == 2
    assert finished == ["done"]


@pytest.mark.asyncio
async def test_inflight_handler_cancelled_on_stop() -> None:
    cancelled: list[bool] = []

    async def handler(_env) -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    h = _hunter(handler)

    async def iter_messages(_ps):
        yield {"type": "message", "channel": b"orion:x", "data": b"x"}
        await asyncio.sleep(0.01)
        h._stop.set()

    h.bus.iter_messages = iter_messages
    await asyncio.wait_for(h._run(), timeout=5)
    assert cancelled == [True]
