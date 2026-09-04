"""Exo Exploration Hub proxy route tests.

Combines two conventions already in use elsewhere in this test dir:
- `test_concept_atlas_routes.py`'s isolated-router `TestClient` pattern (a
  minimal FastAPI app carrying only this router).
- `test_world_pulse_proxy_routes.py`'s direct-call-the-async-handler pattern
  for the degrade path, since that is cheaper than faking an HTTP stack for
  every case.

Covers the graceful-degrade path (not configured / unreachable / bad
status) and the happy path (a mocked aiohttp response), per
`curiosity_routes.py`'s "a dashboard never 500s" contract that this router
was built to mirror.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def _exo_exploration_test_app():
    from scripts.exo_exploration_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client() -> TestClient:
    _ensure_hub_scripts_import_path()
    return TestClient(_exo_exploration_test_app())


# --- degrade path -------------------------------------------------------


def test_finds_degrades_when_base_url_not_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import exo_exploration_routes

    monkeypatch.setattr(
        exo_exploration_routes,
        "_settings",
        lambda: SimpleNamespace(HUB_EXO_EXPLORATION_BASE_URL="", HUB_EXO_EXPLORATION_TIMEOUT_SEC=10.0),
    )
    r = client.get("/api/exo-exploration/finds")
    assert r.status_code == 200  # never a 500 for a broken/absent backend
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "exo_exploration_not_configured"


def test_finds_degrades_when_backend_unreachable(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import exo_exploration_routes

    monkeypatch.setattr(
        exo_exploration_routes,
        "_settings",
        lambda: SimpleNamespace(
            HUB_EXO_EXPLORATION_BASE_URL="http://127.0.0.1:1", HUB_EXO_EXPLORATION_TIMEOUT_SEC=0.2
        ),
    )
    r = client.get("/api/exo-exploration/finds")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "exo_exploration_unreachable"


def test_crawl_runs_degrades_when_not_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import exo_exploration_routes

    monkeypatch.setattr(
        exo_exploration_routes,
        "_settings",
        lambda: SimpleNamespace(HUB_EXO_EXPLORATION_BASE_URL="", HUB_EXO_EXPLORATION_TIMEOUT_SEC=10.0),
    )
    r = client.get("/api/exo-exploration/crawl-runs")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False


# --- happy path (mocked aiohttp) -----------------------------------------


class _FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, get_response: _FakeResponse, calls: list) -> None:
        self._get_response = get_response
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, params=None):
        self._calls.append((url, params))
        return self._get_response


def test_finds_happy_path_forwards_params_and_marks_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import exo_exploration_routes

    monkeypatch.setattr(
        exo_exploration_routes,
        "_settings",
        lambda: SimpleNamespace(
            HUB_EXO_EXPLORATION_BASE_URL="http://orion-exo-exploration:8622",
            HUB_EXO_EXPLORATION_TIMEOUT_SEC=10.0,
        ),
    )
    calls: list = []
    fake_response = _FakeResponse(200, {"finds": [{"title": "RTX 5060", "interest_score": 1.5}], "count": 1})

    class _Session(_FakeSession):
        def __init__(self, *a, **kw):
            super().__init__(fake_response, calls)

    monkeypatch.setattr(exo_exploration_routes.aiohttp, "ClientSession", _Session)

    r = client.get(
        "/api/exo-exploration/finds",
        params={"category": "https://classifieds.ksl.com/search/cat/Computers", "min_interest": 1.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["count"] == 1
    assert body["finds"][0]["title"] == "RTX 5060"

    assert len(calls) == 1
    url, params = calls[0]
    assert url == "http://orion-exo-exploration:8622/finds"
    assert params["category"] == "https://classifieds.ksl.com/search/cat/Computers"
    assert params["min_interest"] == 1.0


def test_crawl_runs_happy_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import exo_exploration_routes

    monkeypatch.setattr(
        exo_exploration_routes,
        "_settings",
        lambda: SimpleNamespace(
            HUB_EXO_EXPLORATION_BASE_URL="http://orion-exo-exploration:8622",
            HUB_EXO_EXPLORATION_TIMEOUT_SEC=10.0,
        ),
    )
    calls: list = []
    fake_response = _FakeResponse(200, {"crawl_runs": [{"status": "success"}], "count": 1})

    class _Session(_FakeSession):
        def __init__(self, *a, **kw):
            super().__init__(fake_response, calls)

    monkeypatch.setattr(exo_exploration_routes.aiohttp, "ClientSession", _Session)

    r = client.get("/api/exo-exploration/crawl-runs")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["crawl_runs"][0]["status"] == "success"
    assert calls[0][0] == "http://orion-exo-exploration:8622/crawl-runs"


def test_finds_degrades_on_non_200_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import exo_exploration_routes

    monkeypatch.setattr(
        exo_exploration_routes,
        "_settings",
        lambda: SimpleNamespace(
            HUB_EXO_EXPLORATION_BASE_URL="http://orion-exo-exploration:8622",
            HUB_EXO_EXPLORATION_TIMEOUT_SEC=10.0,
        ),
    )
    calls: list = []
    fake_response = _FakeResponse(500, {})

    class _Session(_FakeSession):
        def __init__(self, *a, **kw):
            super().__init__(fake_response, calls)

    monkeypatch.setattr(exo_exploration_routes.aiohttp, "ClientSession", _Session)

    r = client.get("/api/exo-exploration/finds")
    assert r.status_code == 200  # still never a 500 out of Hub itself
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "exo_exploration_bad_status"


# --- template / static asset / app.js wiring -----------------------------


def test_index_html_wires_exo_exploration_tab() -> None:
    index_path = HUB_ROOT / "templates" / "index.html"
    index_text = index_path.read_text(encoding="utf-8")
    assert 'id="exoExplorationTabButton"' in index_text
    assert 'data-panel="exo-exploration"' in index_text
    assert 'id="exoExplorationFindsList"' in index_text


def test_exo_exploration_js_file_exists_and_exposes_namespace() -> None:
    js_path = HUB_ROOT / "static" / "js" / "exo-exploration.js"
    assert js_path.is_file()
    js_text = js_path.read_text(encoding="utf-8")
    assert "window.OrionExoExploration" in js_text
    assert "activate" in js_text
    assert "deactivate" in js_text


def test_index_html_loads_exo_exploration_js() -> None:
    index_text = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert "/static/js/exo-exploration.js" in index_text


def test_app_js_pings_activate_and_deactivate_for_exo_exploration() -> None:
    app_js_text = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "exoExplorationPanel" in app_js_text
    assert "OrionExoExploration.activate" in app_js_text
    assert "OrionExoExploration.deactivate" in app_js_text
