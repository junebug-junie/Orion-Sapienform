"""Mesh transport coverage: durable-runs publishes its long-lived rpc_bus's RPC-health window
(the runner's harness/verb RPCs and, since stage 4.5, the GPU pool lease RPCs under the pool's
``gpu_pool_lease`` label) with instance="main". Stage 4.5 deleted every outbound HTTP hop this
service had (gateway /routes, lane /slots, elastic controller, cabinet, thought) together with
app/http_hops.py; no reader keys on those hop names (grep-verified in the 4.5 PR)."""
from __future__ import annotations

from orion.core.bus.async_service import OrionBusAsync


def test_no_outbound_http_hop_module_is_left_behind() -> None:
    import importlib.util

    assert importlib.util.find_spec("app.http_hops") is None
    assert importlib.util.find_spec("app.elastic_runtime") is None


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
