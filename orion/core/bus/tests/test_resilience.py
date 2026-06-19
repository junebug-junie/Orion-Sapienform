from __future__ import annotations

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
