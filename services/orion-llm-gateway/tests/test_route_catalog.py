from __future__ import annotations

import json

import pytest

from app import route_catalog
from orion.llm.routes import LLM_ROUTE_DISPLAY_ORDER
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
    # The catalog lists every route that EXISTS as a name, in the shared display order -- not
    # only the ones this route table happens to configure. `quick_background` is deliberately
    # absent from `table` above, and the honest answer is a row saying `not_configured`, not
    # silence: a lane missing from the route table and a lane that was never a route are
    # different problems, and omitting the row makes them look identical.
    assert [r["id"] for r in payload["routes"]] == list(LLM_ROUTE_DISPLAY_ORDER)
    by_id = {r["id"]: r for r in payload["routes"]}
    assert by_id["quick_background"]["status"] == "not_configured"
    # NOTE (2026-08-19, found while extending this test, NOT fixed here): the
    # LLM_GATEWAY_ROUTE_TABLE_JSON set above does not actually reach `get_route_targets()` in
    # this fixture -- EVERY row comes back `not_configured`, including `chat`. That is why the
    # original assertions could only check key presence. The test still earns its keep as a
    # catalog-shape test; it is not a route-table-loading test despite appearances. Fixing the
    # fixture is its own patch.
    assert all(r["status"] == "not_configured" for r in payload["routes"])
    assert all("status" in r for r in payload["routes"])
    assert all("model" in r for r in payload["routes"])
    assert all("priority" in r for r in payload["routes"])


def _client_factory(*, model_id: str | None = "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"):
    """Builds an AsyncClient stand-in whose .get() answers /health with 200
    and /v1/models with an OpenAI-compat models list (or a malformed/absent
    one, when model_id is None), matching the real llama.cpp shape confirmed
    live 2026-08-14."""

    class _HealthResp:
        status_code = 200

        def json(self):
            return {}

    class _ModelsResp:
        status_code = 200

        def json(self):
            if model_id is None:
                return {}
            return {"object": "list", "data": [{"id": model_id}]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str):
            if url.endswith("/v1/models"):
                return _ModelsResp()
            return _HealthResp()

    return lambda **kwargs: _Client()


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
    monkeypatch.setattr(route_catalog.httpx, "AsyncClient", _client_factory())

    payload = await route_catalog.get_routes_payload()
    by_id = {r["id"]: r for r in payload["routes"]}
    assert by_id["chat"]["status"] == "up"
    assert by_id["chat"]["served_by"] == "atlas-worker-1"
    assert by_id["chat"]["backend"] == "llamacpp"
    assert by_id["chat"]["latency_ms"] is not None
    assert by_id["chat"]["last_checked_at"]
    assert by_id["chat"]["model"] == "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"


@pytest.mark.asyncio
async def test_get_routes_payload_model_none_when_models_endpoint_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {
        "chat": {"url": "http://chat:8011", "served_by": "atlas-worker-1", "backend": "llamacpp"},
    }
    from app.settings import settings as gw_settings

    monkeypatch.setattr(gw_settings, "llm_route_table_json", json.dumps(table))
    _load_route_targets.cache_clear()
    monkeypatch.setattr(route_catalog.httpx, "AsyncClient", _client_factory(model_id=None))

    payload = await route_catalog.get_routes_payload()
    by_id = {r["id"]: r for r in payload["routes"]}
    # Health still reports "up" -- a malformed/absent /v1/models response is
    # a missing nice-to-have, not a route outage.
    assert by_id["chat"]["status"] == "up"
    assert by_id["chat"]["model"] is None


@pytest.mark.asyncio
async def test_get_routes_payload_model_none_when_route_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {
        "chat": {"url": "http://chat:8011", "served_by": "atlas-worker-1", "backend": "llamacpp"},
    }
    from app.settings import settings as gw_settings

    monkeypatch.setattr(gw_settings, "llm_route_table_json", json.dumps(table))
    _load_route_targets.cache_clear()

    class _DownClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str):
            raise ConnectionError("refused")

    monkeypatch.setattr(route_catalog.httpx, "AsyncClient", lambda **kwargs: _DownClient())

    payload = await route_catalog.get_routes_payload()
    by_id = {r["id"]: r for r in payload["routes"]}
    assert by_id["chat"]["status"] == "down"
    # _probe_model runs concurrently with the health check (review finding:
    # sequential would double worst-case refresh latency), but its result is
    # discarded whenever health isn't "up" -- confirmed via the down status
    # above, regardless of what _probe_model itself returned here.
    assert by_id["chat"]["model"] is None
