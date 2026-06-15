from __future__ import annotations

import json

import pytest

from app import route_catalog
from app.llm_backend import RouteTarget, _load_route_targets


@pytest.fixture(autouse=True)
def _clear_route_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _load_route_targets.cache_clear()
    route_catalog.reset_route_health_cache_for_tests()
    monkeypatch.setenv("LLM_ROUTE_DEFAULT", "chat")
    yield
    _load_route_targets.cache_clear()
    route_catalog.reset_route_health_cache_for_tests()


def test_build_routes_response_defaults_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {
        "chat": {"url": "http://chat:8011", "served_by": "atlas-worker-1", "backend": "llamacpp"},
        "quick": {"url": "http://quick:8013", "served_by": "atlas-worker-fast-1", "backend": "llamacpp"},
        "agent": {"url": "http://agent:8014", "served_by": "atlas-worker-agent-1", "backend": "llamacpp"},
        "metacog": {"url": "http://metacog:8012", "served_by": "atlas-worker-2", "backend": "llamacpp"},
    }
    monkeypatch.setenv("LLM_GATEWAY_ROUTE_TABLE_JSON", json.dumps(table))
    _load_route_targets.cache_clear()

    payload = route_catalog.build_routes_response()

    assert payload["default_route"] == "chat"
    assert [r["id"] for r in payload["routes"]] == ["chat", "quick", "agent", "metacog"]
    assert all("status" in r for r in payload["routes"])


@pytest.mark.asyncio
async def test_get_routes_payload_marks_probe_up(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {
        "chat": {"url": "http://chat:8011", "served_by": "atlas-worker-1", "backend": "llamacpp"},
        "quick": {"url": "http://quick:8013", "served_by": "atlas-worker-fast-1", "backend": "llamacpp"},
        "agent": {"url": "http://agent:8014", "served_by": "atlas-worker-agent-1", "backend": "llamacpp"},
        "metacog": {"url": "http://metacog:8012", "served_by": "atlas-worker-2", "backend": "llamacpp"},
    }
    from app.settings import settings as gw_settings

    monkeypatch.setattr(gw_settings, "llm_route_table_json", json.dumps(table))
    _load_route_targets.cache_clear()

    class _Resp:
        status_code = 200

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str):
            return _Resp()

    monkeypatch.setattr(route_catalog.httpx, "AsyncClient", lambda **kwargs: _Client())

    payload = await route_catalog.get_routes_payload()
    by_id = {r["id"]: r for r in payload["routes"]}
    assert by_id["chat"]["status"] == "up"
    assert by_id["chat"]["served_by"] == "atlas-worker-1"
    assert by_id["chat"]["backend"] == "llamacpp"
    assert by_id["chat"]["latency_ms"] is not None
    assert by_id["chat"]["last_checked_at"]
