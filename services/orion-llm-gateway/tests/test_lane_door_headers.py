"""Every dispatch carries the lane door header -- the call's own lease id -- on both the HTTP passthrough and
the bus path. No shared secret is sent (leased-evals design, Phase 2: the door identifies the gateway's calls
by the lease id alone)."""
from __future__ import annotations

import httpx

from app import lane_door, llm_backend, upstream_cancel
from app.settings import settings


def test_headers_carry_only_the_lease():
    assert lane_door.headers("lease-1") == {"X-Orion-Lease-Id": "lease-1"}
    assert lane_door.headers(None) == {}
    assert lane_door.headers("") == {}


def test_the_shared_secret_is_gone_from_the_gateway():
    """Kill means kill: no key setting, no key header, no key-drift check."""
    assert not any("lane_door_key" in name or "lane_door_check" in name for name in type(settings).model_fields)
    assert not hasattr(lane_door, "KEY_HEADER") and not hasattr(lane_door, "observe_grant")


def test_the_bus_path_client_sends_the_lease_of_the_grant_it_runs_under():
    handle = upstream_cancel.UpstreamCancel(lease_id="lease-bus-7")
    seen = {}

    def call():
        with llm_backend._common_http_client(None) as client:
            seen.update(client.headers)

    upstream_cancel.run_cancellable(handle, call)
    assert seen["x-orion-lease-id"] == "lease-bus-7"
    assert "x-orion-lane-key" not in seen


def test_outside_a_grant_the_client_sends_no_lease():
    with llm_backend._common_http_client(None) as client:
        assert "x-orion-lease-id" not in client.headers


def test_the_http_passthrough_sends_its_own_lease_and_not_the_callers(monkeypatch, fake_pool):
    """A caller cannot smuggle a lease id through: only whitelisted request headers are forwarded, and the
    door header comes from the grant."""
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
    assert seen["X-Orion-Lease-Id"] == f"lease-{len(fake_pool.calls)}"   # the grant's id (conftest FakePool)
    assert "x-orion-lane-key" not in {k.lower() for k in seen}
