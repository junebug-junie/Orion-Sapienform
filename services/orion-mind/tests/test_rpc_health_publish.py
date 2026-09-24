"""Mesh transport coverage: each LLM call in app.llm_client._bus_chat opens a short-lived
OrionBusAsync (inside asyncio.run, on a helper Thread when a loop is already running). Its
rpc_request() outcome must fold into the process-wide RPC_HEALTH_SINK before the bus is
closed -- including on timeout -- and app.main's long-lived publisher must drain it."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatResultPayload, ServiceRef
from orion.core.bus.rpc_health_publish import RpcHealthPublisher

INTAKE = "orion:exec:request:LLMGatewayService"


class _CallBus(OrionBusAsync):
    """Real OrionBusAsync (real aggregator + take_rpc_health_aggregator) with the Redis
    round trip replaced: records exactly as rpc_request() does, then replies or times out."""

    instances: list["_CallBus"] = []
    time_out = False

    def __init__(self, *, url: str, enabled: bool = True) -> None:
        super().__init__(url=url, enabled=False)
        self.closed = False
        _CallBus.instances.append(self)

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec, **kw):
        if _CallBus.time_out:
            self._rpc_health.record_timeout(request_channel=request_channel, elapsed_ms=timeout_sec * 1000.0)
            raise TimeoutError("rpc timeout")
        self._rpc_health.record_success(request_channel=request_channel, latency_ms=40.0)
        reply = BaseEnvelope(
            kind="llm.chat.result",
            source=ServiceRef(name="llm-gateway"),
            payload=ChatResultPayload(content='{"ok": true}').model_dump(mode="json"),
        )
        return {"data": self.codec.encode(reply)}


def _call(client) -> None:
    client._bus_chat(
        system_prompt="s", user_prompt="u", route="metacog", options={}, context=None, timeout_sec=5.0
    )


def _drain_publish(main_mod, bus: OrionBusAsync) -> dict:
    bus.publish = AsyncMock()  # type: ignore[method-assign]
    publisher = main_mod.build_rpc_health_publisher(bus)
    assert publisher._kwargs["sinks"] and publisher._connect_bus is True
    # The loop discards whatever the sinks held BEFORE it started (first window = one
    # interval, not "since process start"), so hand this test's already-folded stats to
    # the publish bus the same way the loop's per-tick sink drain does.
    for sink in publisher._kwargs["sinks"]:
        sink.drain_into(bus._rpc_health)

    async def _one_window() -> dict:
        publisher.enabled = True
        publisher._kwargs["interval_sec"] = 0.01
        publisher.start()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if bus.publish.await_count:
                break
        await publisher.stop()
        return bus.publish.await_args_list[0].args[1].payload

    return asyncio.run(_one_window())


def test_per_call_buses_all_land_in_published_window(monkeypatch) -> None:
    import app.llm_client as llm_client
    import app.main as main_mod

    monkeypatch.setattr(llm_client, "OrionBusAsync", _CallBus)
    _CallBus.instances = []
    _CallBus.time_out = False
    llm_client.RPC_HEALTH_SINK.drain_into(OrionBusAsync(url="redis://u:1/0", enabled=False)._rpc_health)
    client = llm_client.MindLLMClient()

    _call(client)  # no running loop -> asyncio.run in this thread

    async def _from_running_loop() -> None:
        _call(client)  # running loop -> helper Thread path (the real /v1/mind/run case)

    asyncio.run(_from_running_loop())

    _CallBus.time_out = True
    with pytest.raises(TimeoutError):
        _call(client)  # timeout is still folded before close

    assert len(_CallBus.instances) == 3
    assert all(b.closed for b in _CallBus.instances)

    payload = _drain_publish(main_mod, OrionBusAsync(url="redis://unused:6379/0", enabled=False))
    assert payload["success_count"] == 2
    assert payload["timeout_count"] == 1
    assert payload["instance"] == "main"
    hop = payload["channel_latency"][INTAKE]
    assert (hop["success_count"], hop["timeout_count"]) == (2, 1)


def test_settings_defaults_publish_on_with_channel_latency(monkeypatch) -> None:
    """Code defaults must equal .env_example (true / 30 / true)."""
    from app.settings import Settings

    for key in ("RPC_HEALTH_PUBLISH_ENABLED", "RPC_HEALTH_PUBLISH_INTERVAL_SEC", "RPC_HEALTH_CHANNEL_LATENCY_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.RPC_HEALTH_PUBLISH_ENABLED is True
    assert s.RPC_HEALTH_PUBLISH_INTERVAL_SEC == 30.0
    assert s.RPC_HEALTH_CHANNEL_LATENCY_ENABLED is True


def test_publisher_wiring_uses_sink() -> None:
    import app.llm_client as llm_client
    import app.main as main_mod

    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    pub = main_mod.build_rpc_health_publisher(bus)
    assert isinstance(pub, RpcHealthPublisher)
    assert pub._kwargs["sinks"] == (llm_client.RPC_HEALTH_SINK,)
    assert pub._kwargs["instance"] == "main"
    assert pub._bus_getter() is bus


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_publisher_bus(monkeypatch) -> None:
    import app.main as main_mod
    from fastapi import FastAPI

    fake_bus = AsyncMock(spec=OrionBusAsync)
    fake_pub = AsyncMock(spec=RpcHealthPublisher)
    monkeypatch.setattr(main_mod, "OrionBusAsync", lambda **kw: fake_bus)
    monkeypatch.setattr(main_mod, "build_rpc_health_publisher", lambda bus: fake_pub)
    monkeypatch.setattr(main_mod, "build_heartbeat_chassis", lambda: AsyncMock())
    monkeypatch.setattr(main_mod.settings, "RPC_HEALTH_PUBLISH_ENABLED", True)
    monkeypatch.setattr(main_mod.settings, "ORION_BUS_ENABLED", True)

    async with main_mod.lifespan(FastAPI()):
        # connect_bus=True: the publisher task connects (with retry), not the lifespan.
        fake_bus.connect.assert_not_awaited()
        fake_pub.start.assert_called_once()
    fake_pub.stop.assert_awaited_once()
    fake_bus.close.assert_awaited_once()
    assert main_mod.rpc_health_bus is None
