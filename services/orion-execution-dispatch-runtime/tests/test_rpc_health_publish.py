"""Mesh transport coverage: every dispatch tick opens its own short-lived OrionBusAsync
(inside asyncio.run on a to_thread worker). Its rpc_request() outcomes must fold into the
process-wide RPC_HEALTH_SINK before the bus is closed, and app.main's long-lived publisher
must drain that sink -- otherwise each tick's stats died with its bus."""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("POSTGRES_URI", "postgresql://unused/unused")

import app.main as dispatch_main
import app.worker as dispatch_worker
from app.settings import Settings
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.rpc_health_publish import RpcHealthPublisher
from orion.execution_dispatch.cortex_client import ExecutionDispatchCortexClient

WORKER_SRC = Path(dispatch_worker.__file__).read_text()
CHANNEL = "orion:cortex:exec:request:background"


class _TickBus(OrionBusAsync):
    """Real OrionBusAsync (real aggregator, real take_rpc_health_aggregator) with the
    Redis round trip replaced: records exactly as rpc_request() does, then replies or
    times out."""

    def __init__(self, *, time_out: bool) -> None:
        super().__init__(url="redis://unused:6379/0", enabled=False)
        self._time_out = time_out
        self.closed = False

    async def connect(self) -> None:  # pragma: no cover - not used
        return None

    async def close(self) -> None:
        self.closed = True

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec, **kw):
        if self._time_out:
            self._rpc_health.record_timeout(request_channel=request_channel, elapsed_ms=timeout_sec * 1000.0)
            raise TimeoutError("rpc timeout")
        self._rpc_health.record_success(request_channel=request_channel, latency_ms=25.0)
        reply = BaseEnvelope(kind="cortex.exec.result", source=ServiceRef(name="x"), payload={"ok": True})
        return {"data": self.codec.encode(reply)}


def _run_tick_in_worker_thread(time_out: bool) -> _TickBus:
    """Mirror of _send_prepared_candidates' bus lifecycle (the AST test below pins that
    the real method has this exact finally shape): fresh bus, real cortex client, the
    real module-level sink, asyncio.run on a non-loop thread."""

    async def _tick() -> _TickBus:
        bus = _TickBus(time_out=time_out)
        client = ExecutionDispatchCortexClient(bus, request_channel=CHANNEL, result_prefix="orion:exec:result")
        try:
            try:
                await client.dispatch(verb="introspect", mode="brain", context={}, dispatch_id="dispatch:t")
            except Exception:
                pass
        finally:
            dispatch_worker.RPC_HEALTH_SINK.absorb_bus(bus)
            await bus.close()
        return bus

    return asyncio.run(_tick())


def test_send_path_absorbs_tick_bus_into_sink_before_close() -> None:
    """Static pin on the real call site: the finally that closes the per-tick bus must
    first fold it into RPC_HEALTH_SINK (a try/finally so a raising/timing-out send is
    still counted)."""
    tree = ast.parse(WORKER_SRC)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_send_prepared_candidates"
    )
    finals = [n.finalbody for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    bodies = [[ast.unparse(stmt) for stmt in fb] for fb in finals]
    assert ["RPC_HEALTH_SINK.absorb_bus(bus)", "await bus.close()"] in bodies


def test_two_tick_buses_both_land_in_the_published_window() -> None:
    dispatch_worker.RPC_HEALTH_SINK.drain_into(OrionBusAsync(url="redis://u:1/0", enabled=False)._rpc_health)
    b1 = asyncio.run(asyncio.to_thread(_run_tick_in_worker_thread, False))
    b2 = asyncio.run(asyncio.to_thread(_run_tick_in_worker_thread, True))  # timeout still counted
    assert b1.closed and b2.closed

    long_lived = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    long_lived.publish = AsyncMock()  # type: ignore[method-assign]
    publisher = dispatch_main.build_rpc_health_publisher(long_lived)
    assert publisher._kwargs["sinks"] and publisher._connect_bus is True
    # The loop discards whatever the sinks held BEFORE it started (first window = one
    # interval, not "since process start"), so hand this test's already-folded stats to
    # the publish bus the same way the loop's per-tick sink drain does.
    for sink in publisher._kwargs["sinks"]:
        sink.drain_into(long_lived._rpc_health)

    async def _one_window() -> dict:
        publisher._kwargs["interval_sec"] = 0.01
        publisher.enabled = True
        publisher.start()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if long_lived.publish.await_count:
                break
        await publisher.stop()
        return long_lived.publish.await_args_list[0].args[1].payload

    payload = asyncio.run(_one_window())
    assert payload["success_count"] == 1
    assert payload["timeout_count"] == 1
    assert payload["instance"] == "main"
    hop = payload["channel_latency"][CHANNEL]
    assert (hop["success_count"], hop["timeout_count"]) == (1, 1)


def test_settings_defaults_publish_on_with_channel_latency(monkeypatch) -> None:
    """Code defaults must equal .env_example (true / 30 / true)."""
    for key in ("RPC_HEALTH_PUBLISH_ENABLED", "RPC_HEALTH_PUBLISH_INTERVAL_SEC", "RPC_HEALTH_CHANNEL_LATENCY_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(POSTGRES_URI="postgresql://u/u")
    assert s.rpc_health_publish_enabled is True
    assert s.rpc_health_publish_interval_sec == 30.0
    assert s.rpc_health_channel_latency_enabled is True


def test_publisher_wiring_uses_sink_and_settings() -> None:
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    pub = dispatch_main.build_rpc_health_publisher(bus)
    assert isinstance(pub, RpcHealthPublisher)
    assert pub._kwargs["sinks"] == (dispatch_worker.RPC_HEALTH_SINK,)
    assert pub._kwargs["instance"] == "main"
    assert pub._bus_getter() is bus


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_publisher_bus(monkeypatch) -> None:
    fake_bus = AsyncMock(spec=OrionBusAsync)
    monkeypatch.setattr(dispatch_main, "OrionBusAsync", lambda **kw: fake_bus)
    fake_pub = AsyncMock(spec=RpcHealthPublisher)
    monkeypatch.setattr(dispatch_main, "build_rpc_health_publisher", lambda bus: fake_pub)
    monkeypatch.setattr(dispatch_main, "build_heartbeat_chassis", lambda: AsyncMock())
    monkeypatch.setattr(dispatch_main.worker, "start", AsyncMock())
    monkeypatch.setattr(dispatch_main.worker, "stop", AsyncMock())
    s = dispatch_main.get_settings()
    monkeypatch.setattr(s, "rpc_health_publish_enabled", True)
    monkeypatch.setattr(s, "orion_bus_enabled", True)

    from fastapi import FastAPI

    async with dispatch_main.lifespan(FastAPI()):
        # connect_bus=True: the publisher task connects (with retry), not the lifespan.
        fake_bus.connect.assert_not_awaited()
        fake_pub.start.assert_called_once()
    fake_pub.stop.assert_awaited_once()
    fake_bus.close.assert_awaited_once()
    assert dispatch_main.rpc_health_bus is None
