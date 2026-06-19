from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name="test-rabbit",
        service_version="0.1.0",
        node_name="node-a",
        bus_url="redis://localhost:6379/0",
        bus_enabled=True,
        heartbeat_interval_sec=0.05,
    )


@pytest.mark.asyncio
async def test_heartbeat_reconnects_and_retries_publish() -> None:
    rabbit = Rabbit(_cfg(), request_channel="orion:test:request", handler=lambda _env: None)
    publish = AsyncMock(side_effect=[RuntimeError("connection lost"), None])
    reconnect = AsyncMock()
    rabbit.bus = SimpleNamespace(publish=publish, reconnect=reconnect)
    rabbit._stop.clear()

    with patch.object(
        rabbit,
        "_source",
        return_value=ServiceRef(name="test-rabbit", node="node-a", version="0.1.0"),
    ), patch(
        "orion.core.bus.bus_service_chassis.asyncio.sleep",
        new=AsyncMock(side_effect=lambda *_: rabbit._stop.set()),
    ):
        await rabbit._heartbeat_loop()

    assert reconnect.await_count == 1
    assert publish.await_count == 2
