"""Mesh transport coverage: durable-runs' outbound HTTP (gateway /routes, lane /slots,
elastic controller) records hops on the long-lived rpc_bus that RpcHealthPublisher
publishes, and the publisher is wired with instance="main"."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from orion.core.bus.async_service import OrionBusAsync

from app import http_hops
from app.admission_runtime import AdmissionRuntime
from app.elastic_runtime import ElasticRuntime


def _mock_inner(monkeypatch, handler):
    monkeypatch.setattr(http_hops.httpx, "AsyncHTTPTransport", lambda *a, **k: httpx.MockTransport(handler))


def _routes_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/routes":
        return httpx.Response(200, json={"routes": [
            {"id": "agent", "upstream": "http://circe:8801", "status": "up"},
        ]})
    if request.url.path == "/slots":
        return httpx.Response(200, json=[{"is_processing": False}])
    return httpx.Response(404)


def _fake_admission(bus):
    return SimpleNamespace(
        settings=SimpleNamespace(lane_policy_json="{}", gateway_url="http://gw:8222", elastic_backend=""),
        hop_recorder_getter=(lambda: bus) if bus is not None else None,
        broker=SimpleNamespace(lanes={}),
        elastic=None,
    )


def test_refresh_lanes_records_gateway_and_slots_hops_on_rpc_bus(monkeypatch) -> None:
    _mock_inner(monkeypatch, _routes_handler)
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    rt = _fake_admission(bus)

    asyncio.run(AdmissionRuntime.refresh_lanes(rt))

    assert rt.broker.lanes["agent"]["healthy"] is True
    hops = bus.get_rpc_health_snapshot().channel_latency
    assert hops["http:gw:8222/routes"].success_count == 1
    assert hops["http:circe:8801/slots"].success_count == 1


def test_refresh_lanes_without_recorder_uses_plain_client(monkeypatch) -> None:
    """No recorder (bus disabled / test runtime): no transport kwarg at all, so the
    client is the pre-instrumentation one (tests that monkeypatch AsyncClient keep working)."""
    assert http_hops.hop_client_kwargs(None) == {}


def test_gateway_timeout_counts_as_hop_timeout(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    _mock_inner(monkeypatch, handler)
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(AdmissionRuntime.refresh_lanes(_fake_admission(bus)))
    hop = bus.get_rpc_health_snapshot().channel_latency["http:gw:8222/routes"]
    assert (hop.success_count, hop.timeout_count) == (0, 1)


def test_elastic_controller_status_504_is_timeout(monkeypatch) -> None:
    """ElasticRuntime reads the recorder from its admission runtime."""
    _mock_inner(monkeypatch, lambda request: httpx.Response(504))
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    runtime = SimpleNamespace(hop_recorder_getter=lambda: bus)
    elastic = ElasticRuntime.__new__(ElasticRuntime)
    elastic.runtime = runtime

    async def go() -> None:
        async with httpx.AsyncClient(timeout=3, **elastic._hop_kwargs()) as client:
            await client.get("http://ctl:8090/v1/gpu-slots/circe-gpu2/status")

    asyncio.run(go())
    hop = bus.get_rpc_health_snapshot().channel_latency["http:ctl:8090/v1/gpu-slots/circe-gpu2/status"]
    assert hop.timeout_count == 1


def test_ids_in_paths_are_normalized(monkeypatch) -> None:
    _mock_inner(monkeypatch, lambda request: httpx.Response(200))
    bus = OrionBusAsync(url="redis://unused:6379/0", enabled=False)

    async def go() -> None:
        async with httpx.AsyncClient(**http_hops.hop_client_kwargs(lambda: bus)) as client:
            await client.get("http://gw:8222/runs/3f2b1c4e-1111-2222-3333-444455556666")
            await client.get("http://gw:8222/runs/aaaaaaaa-1111-2222-3333-444455556666")

    asyncio.run(go())
    hops = bus.get_rpc_health_snapshot().channel_latency
    assert list(hops) == ["http:gw:8222/runs/:id"]
    assert hops["http:gw:8222/runs/:id"].success_count == 2


def test_publisher_wiring_defaults_and_instance(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from app import main
    from app.settings import Settings

    s = Settings()
    assert s.rpc_health_publish_enabled is True
    assert s.rpc_health_channel_latency_enabled is True
    assert s.rpc_health_publish_interval_sec == 30.0

    pub = main.build_rpc_health_publisher()
    assert pub.enabled is (main._settings.rpc_health_publish_enabled and main._settings.orion_bus_enabled)
    assert pub._kwargs["instance"] == "main"
    assert pub._kwargs["include_channel_latency"] == main._settings.rpc_health_channel_latency_enabled
    sentinel = OrionBusAsync(url="redis://unused:6379/0", enabled=False)
    monkeypatch.setattr(main, "rpc_bus", sentinel)
    assert pub._bus_getter() is sentinel
