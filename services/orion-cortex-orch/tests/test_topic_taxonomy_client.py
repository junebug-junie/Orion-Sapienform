from __future__ import annotations

import asyncio

import httpx
import pytest


class _Settings:
    topic_foundry_base_url = "http://topic-foundry:8615"
    topic_foundry_timeout_sec = 3.0
    topic_foundry_labels_cache_ttl_sec = 3600.0


def _reset_cache(mod) -> None:
    mod._cache = None
    mod._cache_at = 0.0


def test_fetch_returns_empty_when_base_url_unset(monkeypatch) -> None:
    import app.topic_taxonomy_client as mod

    _reset_cache(mod)

    class S(_Settings):
        topic_foundry_base_url = ""

    monkeypatch.setattr(mod, "get_settings", lambda: S())
    result = asyncio.run(mod.fetch_active_topic_labels())
    assert result == []


def test_fetch_finds_active_model_run_and_returns_labels(monkeypatch) -> None:
    """End-to-end happy path: /runs items already carry model.stage, so the
    active model's most recent complete run is found in one call, then
    /topics is queried for that run's labels."""
    import app.topic_taxonomy_client as mod

    _reset_cache(mod)
    monkeypatch.setattr(mod, "get_settings", lambda: _Settings())

    runs_response = {
        "items": [
            {"run_id": "run-2", "model": {"stage": "development"}},
            {"run_id": "run-1", "model": {"stage": "active"}},
        ]
    }
    topics_response = {
        "items": [
            {"topic_id": -1, "label": None},
            {"topic_id": 0, "label": "Human-AI Conversations"},
            {"topic_id": 1, "label": "Family Storytime"},
            {"topic_id": 2, "label": None},
        ]
    }

    class _FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            if url.endswith("/runs"):
                assert params["model_name"] == "topic-foundry"
                assert params["status"] == "complete"
                return _FakeResponse(runs_response)
            if url.endswith("/topics"):
                assert params["run_id"] == "run-1"  # the active-stage run, not run-2
                return _FakeResponse(topics_response)
            raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: _FakeClient())
    result = asyncio.run(mod.fetch_active_topic_labels())
    assert result == ["Human-AI Conversations", "Family Storytime"]


def test_fetch_returns_empty_when_no_active_model_run_found(monkeypatch) -> None:
    import app.topic_taxonomy_client as mod

    _reset_cache(mod)
    monkeypatch.setattr(mod, "get_settings", lambda: _Settings())

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": [{"run_id": "run-2", "model": {"stage": "development"}}]}

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            return _FakeResponse()

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: _FakeClient())
    result = asyncio.run(mod.fetch_active_topic_labels())
    assert result == []


def test_fetch_fails_open_on_http_error(monkeypatch) -> None:
    import app.topic_taxonomy_client as mod

    _reset_cache(mod)
    monkeypatch.setattr(mod, "get_settings", lambda: _Settings())

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: _FakeClient())
    result = asyncio.run(mod.fetch_active_topic_labels())
    assert result == []


def test_fetch_caches_within_ttl(monkeypatch) -> None:
    import app.topic_taxonomy_client as mod

    _reset_cache(mod)
    monkeypatch.setattr(mod, "get_settings", lambda: _Settings())

    call_count = 0

    class _FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            nonlocal call_count
            call_count += 1
            if url.endswith("/runs"):
                return _FakeResponse({"items": [{"run_id": "run-1", "model": {"stage": "active"}}]})
            return _FakeResponse({"items": [{"topic_id": 0, "label": "Cat Persona"}]})

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: _FakeClient())

    first = asyncio.run(mod.fetch_active_topic_labels())
    calls_after_first = call_count
    second = asyncio.run(mod.fetch_active_topic_labels())

    assert first == second == ["Cat Persona"]
    assert call_count == calls_after_first, "second call within TTL must not hit the network again"
