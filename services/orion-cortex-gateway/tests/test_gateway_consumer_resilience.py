"""Regression tests for cortex-gateway bus intake consumer resilience."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from orion.core.bus.async_service import OrionBusAsync  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective  # noqa: E402
from app.bus_client import BusClient  # type: ignore  # noqa: E402


class _FakePubSub:
    def __init__(self, broker: "_FakeBroker") -> None:
        self._broker = broker
        self.channels: set[str] = set()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def subscribe(self, *channels: str) -> None:
        for channel in channels:
            self.channels.add(channel)

    async def unsubscribe(self, *channels: str) -> None:
        for channel in channels:
            self.channels.discard(channel)

    async def close(self) -> None:
        self._broker.drop_pubsub(self)

    async def listen(self):
        while True:
            yield await self._queue.get()

    async def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 1.0):
        try:
            msg = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if ignore_subscribe_messages and msg.get("type") != "message":
            return await self.get_message(ignore_subscribe_messages=ignore_subscribe_messages, timeout=timeout)
        return msg

    async def deliver(self, msg: dict[str, Any]) -> None:
        await self._queue.put(msg)


class _FakeCommandRedis:
    def __init__(self, broker: "_FakeBroker") -> None:
        self._broker = broker

    async def ping(self) -> bool:
        return True

    async def publish(self, channel: str, data: bytes) -> None:
        await self._broker.route_publish(channel, data)

    async def close(self) -> None:
        return None

    def pubsub(self) -> _FakePubSub:
        return self._broker.new_pubsub()


class _FakeBroker:
    def __init__(self) -> None:
        self._pubsubs: list[_FakePubSub] = []
        self._command = _FakeCommandRedis(self)

    def command_redis(self) -> _FakeCommandRedis:
        return self._command

    def new_pubsub_redis(self) -> _FakeCommandRedis:
        return _FakeCommandRedis(self)

    def new_pubsub(self) -> _FakePubSub:
        pubsub = _FakePubSub(self)
        self._pubsubs.append(pubsub)
        return pubsub

    def drop_pubsub(self, pubsub: _FakePubSub) -> None:
        with contextlib.suppress(ValueError):
            self._pubsubs.remove(pubsub)

    async def route_publish(self, channel: str, data: bytes) -> None:
        payload = {
            "type": "message",
            "channel": channel.encode("utf-8"),
            "data": data,
        }
        for pubsub in list(self._pubsubs):
            if channel in pubsub.channels:
                await pubsub.deliver(payload)


def _make_fake_bus(broker: _FakeBroker, *, start_rpc_worker: bool = False) -> OrionBusAsync:
    bus = OrionBusAsync("redis://fake", enforce_catalog=False)
    bus._redis = broker.command_redis()
    bus._create_pubsub_redis = broker.new_pubsub_redis  # type: ignore[method-assign]

    async def _connect() -> None:
        if bus._redis is None:
            bus._redis = broker.command_redis()
        await bus._redis.ping()

    bus.connect = _connect  # type: ignore[method-assign]
    if start_rpc_worker:
        bus.start_rpc_worker()
    return bus


def _gateway_bus_message(*, corr_id: str, reply_to: str, prompt: str) -> dict[str, bytes]:
    env = BaseEnvelope(
        kind="cortex.gateway.chat.request",
        source=ServiceRef(name="hub", version="0.4.0", node="test"),
        correlation_id=corr_id,
        reply_to=reply_to,
        payload={"prompt": prompt, "mode": "brain"},
    )
    return {"data": OrionCodec().encode(env)}


def _orch_result_payload(corr_id: str) -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "brain",
        "verb": "chat_general",
        "status": "success",
        "final_text": "ok",
        "memory_used": False,
        "steps": [],
        "correlation_id": corr_id,
        "metadata": {},
        "recall_debug": {},
        "metacog_traces": [],
    }


@pytest.mark.asyncio
async def test_gateway_consumer_dispatches_consecutive_hub_chats() -> None:
    """Regression: intake listener must survive orch RPC and dispatch a second Hub chat."""
    broker = _FakeBroker()
    parent_bus = _make_fake_bus(broker)
    intake_bus = _make_fake_bus(broker)
    rpc_bus = _make_fake_bus(broker, start_rpc_worker=True)

    client = BusClient()
    client.bus = parent_bus
    hub_replies: list[str] = []

    async def _client_connect() -> None:
        await parent_bus.connect()
        client._intake_bus = intake_bus
        await intake_bus.connect()
        client._rpc_bus = rpc_bus
        await rpc_bus.connect()

    client.connect = _client_connect  # type: ignore[method-assign]

    original_rpc_publish = rpc_bus.publish

    async def _orch_auto_reply(channel: str, envelope: BaseEnvelope | dict[str, Any]) -> None:
        await original_rpc_publish(channel, envelope)
        if channel != client.settings.channel_cortex_request:
            return
        corr = str(getattr(envelope, "correlation_id", "") or "")
        reply_ch = f"{client.reply_prefix}:{corr}"
        await asyncio.sleep(0.02)
        orch_env = BaseEnvelope(
            kind="cortex.orch.result",
            source=ServiceRef(name="cortex-orch", version="1", node="test"),
            correlation_id=corr,
            payload=_orch_result_payload(corr),
        )
        await rpc_bus.publish(reply_ch, orch_env)

    rpc_bus.publish = _orch_auto_reply  # type: ignore[method-assign]

    original_parent_publish = parent_bus.publish

    async def _track_hub_replies(channel: str, envelope: BaseEnvelope | dict[str, Any]) -> None:
        await original_parent_publish(channel, envelope)
        if channel.startswith("orion:cortex:gateway:result:"):
            hub_replies.append(channel)

    parent_bus.publish = _track_hub_replies  # type: ignore[method-assign]

    await client.connect()
    await client.start_gateway_consumer()
    await asyncio.sleep(0.05)

    assert client._consumer_task is not None
    assert not client._consumer_task.done()

    for idx in range(2):
        corr = str(uuid4())
        reply_to = f"orion:cortex:gateway:result:{corr}"
        await broker.route_publish(
            client.settings.channel_gateway_request,
            _gateway_bus_message(corr_id=corr, reply_to=reply_to, prompt=f"ping {idx}")["data"],
        )

    deadline = asyncio.get_running_loop().time() + 3.0
    while len(hub_replies) < 2 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)

    assert len(hub_replies) == 2
    assert client._consumer_task is not None
    assert not client._consumer_task.done()

    await client.close()


@pytest.mark.asyncio
async def test_rpc_request_worker_path_skips_ephemeral_subscribe() -> None:
    """Worker-path RPC must not open a throwaway subscribe() context."""
    broker = _FakeBroker()
    bus = _make_fake_bus(broker, start_rpc_worker=True)
    await bus.connect()

    subscribe_calls: list[tuple[str, ...]] = []
    original_subscribe = OrionBusAsync.subscribe

    @asynccontextmanager
    async def _spy_subscribe(self, *channels: str, patterns: bool = False):
        subscribe_calls.append(channels)
        async with original_subscribe(self, *channels, patterns=patterns) as pubsub:
            yield pubsub

    bus.subscribe = _spy_subscribe.__get__(bus, OrionBusAsync)  # type: ignore[method-assign]

    corr = str(uuid4())
    reply_ch = f"orion:cortex:result:{corr}"
    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=ServiceRef(name="cortex-gateway", version="1", node="test"),
        correlation_id=corr,
        reply_to=reply_ch,
        payload=CortexClientRequest(
            mode="brain",
            verb="chat_general",
            packs=["executive_pack"],
            recall=RecallDirective(),
            context=CortexClientContext(
                messages=[],
                raw_user_text="ping",
                user_message="ping",
                session_id="test-session",
                user_id="test-user",
            ),
        ).model_dump(mode="json"),
    )

    async def _reply_later() -> None:
        await asyncio.sleep(0.02)
        reply = BaseEnvelope(
            kind="cortex.orch.result",
            source=ServiceRef(name="cortex-orch", version="1", node="test"),
            correlation_id=corr,
            payload=_orch_result_payload(corr),
        )
        await bus.publish(reply_ch, reply)

    reply_task = asyncio.create_task(_reply_later())
    try:
        msg = await bus.rpc_request(
            "orion:cortex:request",
            env,
            reply_channel=reply_ch,
            timeout_sec=2.0,
        )
    finally:
        await reply_task

    assert msg.get("type") == "message"
    assert subscribe_calls == []
    await bus.close()


@pytest.mark.asyncio
async def test_connect_forks_separate_intake_and_rpc_buses(monkeypatch) -> None:
    """Orch RPC traffic uses a forked bus distinct from the intake subscriber."""
    client = BusClient()
    fork_calls = 0

    async def fake_connect() -> None:
        return None

    async def fake_fork(*, start_rpc_worker: bool = False):
        nonlocal fork_calls
        fork_calls += 1
        child = BusClient()
        child.connect = fake_connect  # type: ignore[method-assign]
        child.close = AsyncMock()
        if start_rpc_worker:
            child._rpc_worker_task = asyncio.get_running_loop().create_future()
        return child

    client.bus = AsyncMock()
    client.bus.connect = fake_connect
    client.bus.fork = fake_fork

    await client.connect()

    assert fork_calls == 2
    assert client._intake_bus is not None
    assert client._rpc_bus is not None
    assert client._intake_bus is not client._rpc_bus
    assert client._intake_bus is not client.bus
    assert client._rpc_bus is not client.bus
    assert client._rpc_bus._rpc_worker_task is not None
