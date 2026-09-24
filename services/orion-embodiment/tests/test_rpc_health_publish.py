"""RPC-health coverage for orion-embodiment: town-speech rpc_request outcomes land on
the worker's long-lived bus, which is the one RpcHealthPublisher drains."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.settings import Settings
from app.worker import EmbodimentWorker
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

SERVICE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_fcc_env(monkeypatch) -> None:
    # EmbodimentWorker() reads ~/.fcc/.env (default /root/.fcc/.env) at construction;
    # unreadable in a sandboxed test run and irrelevant to RPC-health wiring.
    monkeypatch.setattr(EmbodimentWorker, "_load_fcc_env", lambda self: None)


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


def test_settings_defaults_match_env_example_and_compose() -> None:
    fields = Settings.model_fields
    assert fields["rpc_health_publish_enabled"].default is True
    assert fields["rpc_health_publish_interval_sec"].default == 30.0
    assert fields["rpc_health_channel_latency_enabled"].default is True
    env = (SERVICE_ROOT / ".env_example").read_text()
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text()
    for key, val in (
        ("RPC_HEALTH_PUBLISH_ENABLED", "true"),
        ("RPC_HEALTH_PUBLISH_INTERVAL_SEC", "30"),
        ("RPC_HEALTH_CHANNEL_LATENCY_ENABLED", "true"),
    ):
        assert f"{key}={val}" in env
        assert f"{key}=${{{key}:-{val}}}" in compose


def test_worker_publisher_drains_the_worker_bus() -> None:
    w = EmbodimentWorker()
    assert w._rpc_health_publisher._bus_getter() is w._bus
    assert w._rpc_health_publisher._kwargs["instance"] == "main"


@pytest.mark.asyncio
async def test_speech_rpc_outcome_is_published_from_worker_bus() -> None:
    w = EmbodimentWorker()
    bus: OrionBusAsync = w._bus
    fake = _InlineRedis()

    async def _reply(channel, env):
        await fake.pubsub_obj.queue.put({"type": "message", "data": b"{}"})

    env = BaseEnvelope(kind="t", source=ServiceRef(name="t"), payload={})
    with patch.object(bus, "_create_pubsub_redis", return_value=fake):
        bus.publish = _reply  # type: ignore[assignment]
        await bus.rpc_request("orion:cortex:exec:request:chat", env, reply_channel="r1", timeout_sec=1.0)

    bus.publish = AsyncMock()  # type: ignore[assignment]
    pub = w._rpc_health_publisher
    pub.enabled = True
    pub._kwargs["interval_sec"] = 0.01
    pub.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if bus.publish.await_count:
            break
    await pub.stop()
    channel, published = bus.publish.await_args_list[0].args
    assert channel == "orion:rpc_health:snapshot"
    assert published.payload["channel_latency"]["orion:cortex:exec:request:chat"]["success_count"] == 1


@pytest.mark.asyncio
async def test_start_and_stop_drive_the_publisher() -> None:
    w = EmbodimentWorker()
    w._settings = w._settings.model_copy(update={"enabled": True, "perception_interval_sec": 0})
    w._bus.connect = AsyncMock()  # type: ignore[method-assign]
    w._bus.close = AsyncMock()  # type: ignore[method-assign]
    w._consume_loop = AsyncMock()  # type: ignore[method-assign]
    started = []
    w._rpc_health_publisher.start = lambda: started.append(True)  # type: ignore[method-assign]
    w._rpc_health_publisher.stop = AsyncMock()  # type: ignore[method-assign]
    await w.start()
    await w.stop()
    assert started == [True]
    w._rpc_health_publisher.stop.assert_awaited_once()
