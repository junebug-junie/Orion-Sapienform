"""Every dispatch carries the lane door headers -- the key and the call's own lease id -- on both the
HTTP passthrough and the bus path; the gateway checks off the call path that each door holds its key."""
from __future__ import annotations

import logging

import httpx
import pytest

from app import lane_door, llm_backend, upstream_cancel
from app.settings import settings

KEY = "g" * 64


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_lane_door_key", KEY)
    lane_door._reset_for_tests()
    yield
    lane_door._reset_for_tests()


def test_headers_carry_the_key_and_the_lease(key):
    assert lane_door.headers("lease-1") == {"X-Orion-Lane-Key": KEY, "X-Orion-Lease-Id": "lease-1"}


def test_no_key_configured_sends_only_the_lease(monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_lane_door_key", "")
    assert lane_door.headers("lease-1") == {"X-Orion-Lease-Id": "lease-1"}
    assert lane_door.headers(None) == {}


def test_the_bus_path_client_sends_the_headers_of_the_grant_it_runs_under(key):
    handle = upstream_cancel.UpstreamCancel(lease_id="lease-bus-7")
    seen = {}

    def call():
        with llm_backend._common_http_client(None) as client:
            seen.update(client.headers)

    upstream_cancel.run_cancellable(handle, call)
    assert seen["x-orion-lane-key"] == KEY and seen["x-orion-lease-id"] == "lease-bus-7"


def test_outside_a_grant_the_client_sends_no_lease(key):
    with llm_backend._common_http_client(None) as client:
        assert "x-orion-lease-id" not in client.headers


def _stats(body, status=200):
    return httpx.MockTransport(lambda req: httpx.Response(status, json=body))


@pytest.mark.parametrize("body,status,state,level", [
    ({"lane": "chat", "mode": "shadow", "key_hash": lane_door.key_hash(KEY)}, 200, "ok", logging.INFO),
    ({"lane": "chat", "mode": "shadow", "key_hash": "0" * 16}, 200, "mismatch", logging.WARNING),
    ({"lane": "chat", "mode": "shadow", "key_hash": None}, 200, "mismatch", logging.WARNING),
    ({"error": "not found"}, 404, "absent", logging.INFO),
])
def test_check_compares_key_hashes(key, monkeypatch, caplog, body, status, state, level):
    real_get = httpx.get
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Client(transport=_stats(body, status)).get(url))
    with caplog.at_level(logging.INFO, logger="orion-llm-gateway.lane-door"):
        result = lane_door.check_now("http://circe.test:8011")
    assert result["state"] == state
    assert any(r.levelno == level and f"lane_door_" in r.getMessage() for r in caplog.records)
    assert KEY not in caplog.text                         # never the key itself
    assert lane_door.status()["servers"]["http://circe.test:8011"]["state"] == state
    httpx.get = real_get


def test_observe_grant_checks_each_server_once_per_interval(key, monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_lane_door_check_enabled", True)
    monkeypatch.setattr(settings, "llm_gateway_lane_door_check_sec", 600.0)
    started = []

    class T:
        def __init__(self, target, args, daemon, name):
            started.append(args)

        def start(self):
            pass

    monkeypatch.setattr(lane_door.threading, "Thread", T)
    for _ in range(3):
        lane_door.observe_grant("http://circe.test:8011/v1/chat/completions")
    lane_door.observe_grant("http://circe.test:8013")
    assert started == [("http://circe.test:8011",), ("http://circe.test:8013",)]


def test_observe_grant_does_nothing_when_disabled(key, monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_lane_door_check_enabled", False)
    lane_door.observe_grant("http://circe.test:8011")
    assert lane_door.status()["servers"] == {}


def test_the_http_passthrough_sends_the_key_and_its_own_lease_and_not_the_callers(key, monkeypatch, fake_pool):
    """A caller cannot smuggle its own lane key through: only whitelisted request headers are forwarded,
    and the door headers come from the gateway's settings and the grant."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    seen = {}

    async def fake_post(self, url, **kwargs):
        seen.update(kwargs["headers"])
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    r = TestClient(app).post("/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]},
                             headers={"X-Orion-Lane-Key": "caller-guess", "X-Orion-Lease-Id": "caller-lease"})
    assert r.status_code == 200
    assert seen["X-Orion-Lane-Key"] == KEY
    assert seen["X-Orion-Lease-Id"] == f"lease-{len(fake_pool.calls)}"   # the grant's id (conftest FakePool)


def test_both_sides_without_a_key_is_unkeyed_not_a_mismatch(monkeypatch, caplog):
    monkeypatch.setattr(settings, "llm_gateway_lane_door_key", "")
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Client(
        transport=_stats({"lane": "chat", "mode": "shadow", "key_hash": None})).get(url))
    with caplog.at_level(logging.INFO, logger="orion-llm-gateway.lane-door"):
        assert lane_door.check_now("http://circe.test:8011")["state"] == "unkeyed"
    assert "lane_door_key_mismatch" not in caplog.text
    lane_door._reset_for_tests()
