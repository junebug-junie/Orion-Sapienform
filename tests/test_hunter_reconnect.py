from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name="test-hunter",
        service_version="0",
        node_name="node-a",
        bus_url="redis://localhost:6379/0",
        bus_enabled=True,
    )


@pytest.mark.asyncio
async def test_hunter_reconnects_after_subscribe_failure() -> None:
  attempts = {"count": 0}
  handled: list[str] = []

  async def handler(env: BaseEnvelope) -> None:
    handled.append(env.kind)

  hunter = Hunter(_cfg(), handler=handler, patterns=["orion:equilibrium:metacog:trigger"])
  hunter.bus = MagicMock()
  hunter.bus.enabled = True
  hunter.bus.redis = object()
  hunter.bus.connect = AsyncMock()
  hunter.bus.codec.decode = MagicMock()

  @asynccontextmanager
  async def subscribe_ctx(*_channels, patterns: bool = False):
    attempts["count"] += 1
    if attempts["count"] == 1:
      raise ConnectionError("pubsub dropped")
    pubsub = MagicMock()
    yield pubsub

  async def iter_messages(_pubsub):
    env = BaseEnvelope(
      kind="orion.metacog.trigger.v1",
      source=ServiceRef(name="equilibrium", node="n1"),
      payload={},
    )
    yield {"type": "message", "channel": b"orion:equilibrium:metacog:trigger", "data": b"x"}
    hunter._stop.set()
    return

  hunter.bus.subscribe = subscribe_ctx
  hunter.bus.iter_messages = iter_messages
  hunter.bus.codec.decode.return_value = MagicMock(ok=True, envelope=BaseEnvelope(
    kind="orion.metacog.trigger.v1",
    source=ServiceRef(name="equilibrium", node="n1"),
    payload={},
  ))
  hunter._publish_error = AsyncMock()

  await hunter._run()

  assert attempts["count"] >= 2
  assert handled == ["orion.metacog.trigger.v1"]
