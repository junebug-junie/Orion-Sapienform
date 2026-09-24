"""RPC-health coverage for orion-actions: every rpc_request runs on the long-lived RPC
bus fork, and that same bus is what RpcHealthPublisher drains and publishes."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import main as actions_main
from app.settings import Settings

SERVICE_ROOT = Path(__file__).resolve().parents[1]


class _InlinePubSub:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    async def subscribe(self, *channels: str) -> None:
        return None

    async def unsubscribe(self, *channels: str) -> None:
        return None

    async def close(self) -> None:
        return None

    async def listen(self):
        while True:
            yield await self.queue.get()


class _InlineRedis:
    def __init__(self) -> None:
        self.pubsub_obj = _InlinePubSub()

    def pubsub(self):
        return self.pubsub_obj

    async def close(self) -> None:
        return None


def _factory(reply_channel: str, attempt: int) -> BaseEnvelope:
    return BaseEnvelope(
        kind="test.request",
        source=ServiceRef(name="test", version="1"),
        correlation_id=str(uuid4()),
        reply_to=reply_channel,
        payload={"attempt": attempt},
    )


def test_settings_defaults_match_env_example() -> None:
    fields = Settings.model_fields
    assert fields["rpc_health_publish_enabled"].default is True
    assert fields["rpc_health_publish_interval_sec"].default == 30.0
    assert fields["rpc_health_channel_latency_enabled"].default is True
    env = (SERVICE_ROOT / ".env_example").read_text()
    for line in (
        "RPC_HEALTH_PUBLISH_ENABLED=true",
        "RPC_HEALTH_PUBLISH_INTERVAL_SEC=30",
        "RPC_HEALTH_CHANNEL_LATENCY_ENABLED=true",
    ):
        assert line in env


@pytest.mark.asyncio
async def test_rpc_outcomes_on_rpc_bus_are_published(monkeypatch) -> None:
    monkeypatch.setattr(actions_main.settings, "orion_bus_enabled", True)  # conftest disables it
    bus = OrionBusAsync("redis://unused:6379/0")
    fake = _InlineRedis()

    async def _reply(channel, env):
        await fake.pubsub_obj.queue.put({"type": "message", "data": b"{}"})

    with patch.object(bus, "_create_pubsub_redis", return_value=fake):
        bus.publish = _reply  # type: ignore[assignment]
        await actions_main._rpc_request_with_retry(
            bus=bus,
            request_channel="orion:cortex:exec:request:background",
            reply_prefix="orion:exec:result",
            timeout_sec=1.0,
            envelope_factory=_factory,
            operation_name="t",
            max_attempts=1,
        )

    bus.publish = AsyncMock()  # type: ignore[assignment]
    pub = actions_main.build_rpc_health_publisher(lambda: bus)
    assert pub.enabled is True
    pub._kwargs["interval_sec"] = 0.01
    pub.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if bus.publish.await_count:
            break
    await pub.stop()
    channel, env = bus.publish.await_args_list[0].args
    assert channel == "orion:rpc_health:snapshot"
    assert env.payload["instance"] == "main"
    assert env.payload["success_count"] == 1
    assert env.payload["channel_latency"]["orion:cortex:exec:request:background"]["success_count"] == 1


def test_publisher_disabled_when_bus_disabled(monkeypatch) -> None:
    monkeypatch.setattr(actions_main.settings, "orion_bus_enabled", False)
    assert actions_main.build_rpc_health_publisher(lambda: None).enabled is False


def test_lifespan_publishes_the_rpc_fork_and_stops_it() -> None:
    """Static guard (the lifespan boots a real Hunter): the publisher must drain the
    RPC fork every rpc_request uses, not the Hunter's listener bus."""
    src = (SERVICE_ROOT / "app" / "main.py").read_text()
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "build_rpc_health_publisher"
    ]
    assert [ast.unparse(c.args[0]) for c in calls] == ["lambda: _actions_rpc_bus"]
    assert "await app.state.rpc_health_publisher.stop()" in src
