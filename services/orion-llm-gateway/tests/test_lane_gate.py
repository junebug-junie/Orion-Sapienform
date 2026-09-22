"""The operator gate on a lent lane fails closed and is the only thing that opens `chat-burst`."""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from app import lane_gate, route_catalog
from app.capacity import CapacityPermit, CapacityRejected
from app.llm_backend import _load_route_targets
from app.main import app
from app.settings import settings


class FakeRedis:
    def __init__(self, *, broken: bool = False) -> None:
        self.store: Dict[str, str] = {}
        self.broken = broken

    async def get(self, key: str):
        if self.broken:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key: str, value: str):
        if self.broken:
            raise ConnectionError("redis down")
        self.store[key] = value


@pytest.fixture
def fake_redis():
    client = FakeRedis()
    lane_gate.reset_for_tests(client)
    yield client
    lane_gate.reset_for_tests(None)


@pytest.mark.asyncio
async def test_missing_key_is_closed(fake_redis: FakeRedis) -> None:
    state = await lane_gate.read_gate("chat-burst")
    assert state == {"route_id": "chat-burst", "open": False, "changed_at": None,
                     "changed_by": None, "lends_route": "chat"}
    assert await lane_gate.is_open("chat-burst") is False


@pytest.mark.asyncio
async def test_set_then_read_round_trips_and_records_who(fake_redis: FakeRedis) -> None:
    opened = await lane_gate.set_gate("chat-burst", open=True, changed_by="hub-ui")
    assert opened["open"] is True and opened["changed_by"] == "hub-ui" and opened["changed_at"]
    assert json.loads(fake_redis.store[lane_gate.gate_key("chat-burst")])["open"] is True
    assert await lane_gate.is_open("chat-burst") is True
    closed = await lane_gate.set_gate("chat-burst", open=False, changed_by="hub-ui")
    assert closed["open"] is False
    assert await lane_gate.is_open("chat-burst") is False


@pytest.mark.asyncio
async def test_redis_failure_reads_as_closed_never_open() -> None:
    lane_gate.reset_for_tests(FakeRedis(broken=True))
    try:
        with pytest.raises(lane_gate.LaneGateUnavailable):
            await lane_gate.read_gate("chat-burst")
        assert await lane_gate.is_open("chat-burst") is False
    finally:
        lane_gate.reset_for_tests(None)


@pytest.mark.asyncio
async def test_corrupt_value_reads_as_closed(fake_redis: FakeRedis) -> None:
    fake_redis.store[lane_gate.gate_key("chat-burst")] = "not json"
    assert await lane_gate.is_open("chat-burst") is False


@pytest.mark.asyncio
async def test_only_operator_gated_routes_have_a_gate(fake_redis: FakeRedis) -> None:
    for route_id in ("chat", "agent-burst", "harness", "nope"):
        with pytest.raises(lane_gate.RouteNotOperatorGated):
            await lane_gate.read_gate(route_id)
        with pytest.raises(lane_gate.RouteNotOperatorGated):
            await lane_gate.set_gate(route_id, open=True, changed_by="x")


def test_gate_endpoints_round_trip_and_refuse_ungated_routes(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis) -> None:
    # No route table configured: the gate is state, not a probe, and must work regardless.
    monkeypatch.setattr(settings, "llm_route_table_json", "{}")
    _load_route_targets.cache_clear()

    async def _no_refresh(*, force: bool = False) -> None:
        return None

    monkeypatch.setattr("app.main.refresh_route_health_cache", _no_refresh)
    client = TestClient(app)
    assert client.get("/routes/chat-burst/gate").json()["open"] is False
    resp = client.put("/routes/chat-burst/gate", json={"open": True, "changed_by": "hub-ui"})
    assert resp.status_code == 200 and resp.json()["open"] is True
    assert client.get("/routes/chat-burst/gate").json()["changed_by"] == "hub-ui"
    assert client.get("/routes/chat/gate").status_code == 404
    assert client.put("/routes/agent-burst/gate", json={"open": True}).status_code == 404


@pytest.mark.asyncio
async def test_chat_burst_refuses_unleased_calls_like_agent_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    for enabled in (False, True):
        monkeypatch.setattr(settings, "llm_gateway_capacity_enabled", enabled)
        ticket = CapacityPermit(lane="chat-burst", backend_key="http://chat", correlation_id="t", budget_sec=1)
        with pytest.raises(CapacityRejected, match="chat_burst_requires_durable_capacity_lease"):
            await ticket.acquire()


@pytest.mark.asyncio
async def test_catalog_reports_operator_closed_until_gate_opens(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis) -> None:
    table = {
        "chat": {"url": "http://chat:8011", "served_by": "circe-worker-1", "backend": "llamacpp"},
        "chat-burst": {"url": "http://chat:8011", "served_by": "circe-worker-1", "backend": "llamacpp", "priority": "system"},
    }
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(table))
    _load_route_targets.cache_clear()

    async def _probe(target):  # worker is UP
        return "up", 5, "some-model", False, 65536

    monkeypatch.setattr(route_catalog, "_probe_backend", _probe)
    route_catalog._cache.clear()
    await route_catalog.refresh_route_health_cache(force=True)
    by_id = {r["id"]: r for r in route_catalog.build_routes_response()["routes"]}
    assert by_id["chat"]["status"] == "up" and by_id["chat"]["gate_open"] is None
    assert by_id["chat-burst"]["status"] == "operator_closed"
    assert by_id["chat-burst"]["gate_open"] is False
    assert by_id["chat-burst"]["priority"] == "system"

    await lane_gate.set_gate("chat-burst", open=True, changed_by="test")
    await route_catalog.refresh_route_health_cache(force=True)
    by_id = {r["id"]: r for r in route_catalog.build_routes_response()["routes"]}
    assert by_id["chat-burst"]["status"] == "up" and by_id["chat-burst"]["gate_open"] is True
    route_catalog._cache.clear()


@pytest.mark.asyncio
async def test_catalog_open_gate_does_not_resurrect_a_down_worker(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis) -> None:
    table = {"chat-burst": {"url": "http://chat:8011", "served_by": "circe-worker-1", "backend": "llamacpp"}}
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(table))
    _load_route_targets.cache_clear()

    async def _probe(target):
        return "down", None, None, None, None

    monkeypatch.setattr(route_catalog, "_probe_backend", _probe)
    await lane_gate.set_gate("chat-burst", open=True, changed_by="test")
    route_catalog._cache.clear()
    await route_catalog.refresh_route_health_cache(force=True)
    entry = {r["id"]: r for r in route_catalog.build_routes_response()["routes"]}["chat-burst"]
    assert entry["status"] == "down" and entry["gate_open"] is True
    route_catalog._cache.clear()


def test_passthrough_refuses_closed_gate_before_touching_the_worker(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis) -> None:
    table = {"chat-burst": {"url": "http://chat:8011", "served_by": "circe-worker-1", "backend": "llamacpp"}}
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(table))
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True, raising=False)
    _load_route_targets.cache_clear()
    client = TestClient(app)
    body: Dict[str, Any] = {"model": "llamacpp/chat-burst", "max_tokens": 8,
                            "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "route_operator_closed"
    resp = client.post("/v1/chat/completions", json={"model": "llamacpp/chat-burst",
                                                     "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "route_operator_closed"
