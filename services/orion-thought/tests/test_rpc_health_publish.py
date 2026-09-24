"""orion-thought RPC-health coverage: every RPC/HTTP hop lands in the process sink that
the one published window drains (app/rpc_health.py) -- per-call buses and long-lived
worker buses alike -- and the publisher is wired into the lifespan."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.rpc_health import RpcHealthAggregator

SERVICE_ROOT = Path(__file__).resolve().parents[1]
SRC = ServiceRef(name="orion-thought", version="0", node="athena")


def _drain_sink() -> RpcHealthAggregator:
    from app.rpc_health import RPC_HEALTH_SINK

    agg = RpcHealthAggregator()
    RPC_HEALTH_SINK.drain_into(agg)
    return agg


@pytest.fixture(autouse=True)
def _clean_sink():
    _drain_sink()
    yield
    _drain_sink()


def _driven_bus(*, reply_payload: dict | None) -> OrionBusAsync:
    """A real OrionBusAsync whose worker-path rpc_request runs for real (so its own
    RPC-health recording is exercised), with Redis replaced: publish() answers the
    pending future with ``reply_payload`` (or never answers when None -> timeout)."""
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=True)
    bus._rpc_worker_task = asyncio.get_running_loop().create_future()  # "worker running"
    bus._rpc_subscribe = AsyncMock()  # type: ignore[method-assign]
    bus._emit_rpc_timeout_grammar = AsyncMock()  # type: ignore[method-assign]

    async def _publish(channel: str, env: BaseEnvelope) -> None:
        if reply_payload is None:
            return
        reply = BaseEnvelope(kind="reply", source=SRC, correlation_id=env.correlation_id, payload=reply_payload)
        bus._pending_rpc[(env.reply_to, str(env.correlation_id))].set_result({"data": bus.codec.encode(reply)})

    bus.publish = _publish  # type: ignore[method-assign]
    return bus


def _plan_request():
    from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest

    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="reverie_narrate", steps=[]),
        args=PlanExecutionArgs(request_id="r1"),
        context={},
    )


def _env_example() -> dict[str, str]:
    out = {}
    for line in (SERVICE_ROOT / ".env_example").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def test_settings_defaults_match_env_example() -> None:
    from app.settings import ThoughtSettings

    fields = ThoughtSettings.model_fields
    env = _env_example()
    assert fields["rpc_health_publish_enabled"].default is True
    assert fields["rpc_health_publish_interval_sec"].default == 30.0
    assert fields["rpc_health_channel_latency_enabled"].default is True
    assert env["RPC_HEALTH_PUBLISH_ENABLED"] == "true"
    assert float(env["RPC_HEALTH_PUBLISH_INTERVAL_SEC"]) == 30.0
    assert env["RPC_HEALTH_CHANNEL_LATENCY_ENABLED"] == "true"
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text()
    for key, default in (
        ("RPC_HEALTH_PUBLISH_ENABLED", "true"),
        ("RPC_HEALTH_PUBLISH_INTERVAL_SEC", "30"),
        ("RPC_HEALTH_CHANNEL_LATENCY_ENABLED", "true"),
    ):
        assert f"{key}=${{{key}:-{default}}}" in compose


@pytest.mark.asyncio
async def test_cortex_exec_rpc_lands_in_published_window() -> None:
    from app.cortex_client import CortexExecClient
    from app.rpc_health import build_publisher, fold_bus

    bus = _driven_bus(reply_payload={"result": {"ok": True}})
    client = CortexExecClient(bus, request_channel="orion:cortex:exec:request:background", result_prefix="orion:exec:result")
    out = await client.execute_plan(source=SRC, req=_plan_request(), correlation_id=str(uuid4()), timeout_sec=5.0)
    assert out == {"ok": True}
    fold_bus(bus)  # what every thought bus does per tick / in finally

    pub_bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    pub_bus.publish = AsyncMock()  # type: ignore[method-assign]
    settings = SimpleNamespace(
        rpc_health_publish_enabled=True,
        orion_bus_enabled=True,
        service_name="orion-thought",
        service_version="0",
        node_name="athena",
        rpc_health_publish_interval_sec=0.01,
        rpc_health_channel_latency_enabled=True,
    )
    publisher = build_publisher(settings, lambda: pub_bus)
    assert publisher._kwargs["sinks"] and publisher._connect_bus is True
    # The loop discards whatever the sinks held BEFORE it started (first window = one
    # interval, not "since process start"), so hand this test's already-folded stats to
    # the publish bus the same way the loop's per-tick sink drain does.
    for sink in publisher._kwargs["sinks"]:
        sink.drain_into(pub_bus._rpc_health)
    publisher.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if pub_bus.publish.await_count:
            break
    await publisher.stop()

    payload = pub_bus.publish.await_args_list[0].args[1].payload
    assert payload["instance"] == "main"
    assert payload["success_count"] == 1
    hop = payload["channel_latency"]["orion:cortex:exec:request:background"]
    assert (hop["success_count"], hop["timeout_count"]) == (1, 0)


@pytest.mark.asyncio
async def test_vision_caption_rpc_timeout_counts_as_timeout(monkeypatch) -> None:
    from app import visual_chain
    from app.rpc_health import fold_bus

    monkeypatch.setattr(visual_chain.settings, "channel_vision_host_request", "orion:exec:request:VisionHostService:circe-vl")
    bus = _driven_bus(reply_payload=None)
    assert await visual_chain.request_caption(bus, "a" * 64, timeout_sec=0.05) is None  # fail-open
    fold_bus(bus)
    snap = _drain_sink().snapshot_and_reset()
    hop = snap.channel_latency["orion:exec:request:VisionHostService:circe-vl"]
    assert (hop.success_count, hop.timeout_count) == (0, 1)


@pytest.mark.asyncio
async def test_mind_http_hop_success_504_and_timeout(monkeypatch) -> None:
    from app import mind_enrichment
    from orion.mind.v1 import MindRunRequestV1

    outcomes = iter(["ok", "504", "timeout"])

    def _handler(request: httpx.Request) -> httpx.Response:
        kind = next(outcomes)
        if kind == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(504 if kind == "504" else 200, json={})

    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: httpx.MockTransport(_handler))
    settings = SimpleNamespace(mind_base_url="http://mind.test:6611", mind_timeout_sec=5.0)
    req = MindRunRequestV1.model_validate({"correlation_id": str(uuid4()), "user_text": "hi"})
    for _ in range(3):
        # fail-open: every outcome (bad body, 504, timeout) returns None, never raises
        assert await mind_enrichment.run_mind_for_thought(req, settings=settings, correlation_id="c") is None

    hop = _drain_sink().snapshot_and_reset().channel_latency["http:mind.test:6611/v1/mind/run"]
    assert (hop.success_count, hop.timeout_count) == (1, 2)


@pytest.mark.asyncio
async def test_per_call_handler_bus_is_folded(monkeypatch) -> None:
    from app import bus_listener

    class _Bus(OrionBusAsync):
        async def connect(self):  # type: ignore[override]
            return None

        async def close(self):  # type: ignore[override]
            return None

    async def _handle(bus, raw_msg):
        bus._rpc_health.record_timeout(request_channel="orion:cortex:exec:request", elapsed_ms=9000.0)
        raise RuntimeError("handler blew up after the RPC")  # fold must still happen

    monkeypatch.setattr(bus_listener, "OrionBusAsync", _Bus)
    monkeypatch.setattr(bus_listener, "_handle_bus_message", _handle)
    await bus_listener._run_bus_message_handler({"data": b""})

    snap = _drain_sink().snapshot_and_reset()
    assert snap.timeout_count == 1
    assert snap.channel_latency["orion:cortex:exec:request"].timeout_count == 1


@pytest.mark.asyncio
async def test_long_lived_reverie_worker_folds_every_tick(monkeypatch) -> None:
    from app import reverie

    class _Bus(OrionBusAsync):
        async def connect(self):  # type: ignore[override]
            return None

        async def close(self):  # type: ignore[override]
            return None

    stop = asyncio.Event()
    ticks = {"n": 0}

    async def _once(bus, **_kw):
        ticks["n"] += 1
        bus._rpc_health.record_success(request_channel="orion:cortex:exec:request", latency_ms=10.0)
        if ticks["n"] == 1:
            # Before the worker exits: tick 1 must already be in the sink, not held on
            # the worker's bus until process shutdown.
            return None
        stop.set()

    monkeypatch.setattr(reverie, "OrionBusAsync", _Bus)
    monkeypatch.setattr(reverie, "run_reverie_once", _once)
    monkeypatch.setattr(reverie.settings, "reverie_enabled", True)
    monkeypatch.setattr(reverie.settings, "reverie_chain_enabled", False)
    monkeypatch.setattr(reverie.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(reverie.settings, "reverie_interval_sec", 0.01)

    worker = asyncio.create_task(reverie.run_reverie_worker(stop))
    for _ in range(200):
        await asyncio.sleep(0.005)
        if ticks["n"] >= 1:
            break
    await asyncio.sleep(0.001)
    first = _drain_sink().snapshot_and_reset()
    assert first.success_count >= 1  # folded per tick
    await asyncio.wait_for(worker, timeout=2.0)


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_publisher(monkeypatch) -> None:
    import orion.core.bus.async_service as async_service_mod
    import app.main as main_module
    from app.settings import settings

    async def _noop(*_a, **_k):
        return None

    for name in (
        "run_bus_worker",
        "run_reverie_worker",
        "run_reverie_chain_worker",
        "run_reasoning_worker",
        "run_visual_chain_worker",
        "run_visual_chain_watchdog",
        "warm_pool",
    ):
        monkeypatch.setattr(main_module, name, _noop)
    monkeypatch.setattr(main_module, "build_heartbeat_chassis", lambda: (_ for _ in ()).throw(RuntimeError("no hb")))

    made: list[OrionBusAsync] = []

    class _Bus(OrionBusAsync):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

        async def connect(self):  # type: ignore[override]
            self.connected = True

        async def close(self):  # type: ignore[override]
            self.closed = True

    monkeypatch.setattr(async_service_mod, "OrionBusAsync", _Bus)
    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    monkeypatch.setattr(settings, "rpc_health_publish_enabled", True)

    app = SimpleNamespace(state=SimpleNamespace())
    async with main_module.lifespan(app):
        assert app.state.rpc_health_publisher.running
        assert app.state.rpc_health_bus is made[0]
        for _ in range(100):  # the publisher task connects it (connect_bus=True)
            if getattr(made[0], "connected", False):
                break
            await asyncio.sleep(0.01)
        assert getattr(made[0], "connected", False) is True
    assert not app.state.rpc_health_publisher.running
    assert getattr(made[0], "closed", False) is True

    # Disabled -> no bus, no task.
    monkeypatch.setattr(settings, "rpc_health_publish_enabled", False)
    made.clear()
    app = SimpleNamespace(state=SimpleNamespace())
    async with main_module.lifespan(app):
        assert not app.state.rpc_health_publisher.running
        assert app.state.rpc_health_bus is None
    assert made == []


# Subscribe-only loops never call rpc_request, so there is nothing to fold; the lifespan's
# bus IS the publisher bus (drained directly by get_rpc_health_snapshot).
_SUBSCRIBE_ONLY_BUS_FUNCS = {
    ("bus_listener.py", "run_bus_worker"),
    ("reasoning_activity.py", "run_reasoning_worker"),
    ("main.py", "lifespan"),
}


def test_every_rpc_bus_in_app_folds_into_the_sink() -> None:
    """Static guard: a function that constructs an OrionBusAsync must fold it
    (app/rpc_health.fold_bus), or its RPC-health window is silently discarded. Covers
    the reverie-chain / visual-chain workers and /visual-chain/run-once, which the
    behavioural tests above don't drive, and any bus added later."""
    import ast

    missing = []
    for path in sorted((SERVICE_ROOT / "app").glob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            names = {getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in calls}
            if "OrionBusAsync" in names and "fold_bus" not in names:
                if (path.name, fn.name) not in _SUBSCRIBE_ONLY_BUS_FUNCS:
                    missing.append(f"{path.name}:{fn.name}")
    assert missing == []
