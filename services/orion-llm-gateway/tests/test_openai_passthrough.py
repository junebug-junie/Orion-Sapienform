from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app import openai_passthrough
from app.llm_backend import _load_route_targets
from app.main import app
from app.settings import settings


@pytest.fixture(autouse=True)
def _clear_route_cache() -> None:
    _load_route_targets.cache_clear()
    yield
    _load_route_targets.cache_clear()


@pytest.fixture
def route_table() -> Dict[str, Dict[str, str]]:
    return {
        "chat": {"url": "http://chat:8011", "served_by": "atlas-worker-1", "backend": "llamacpp"},
        "quick": {"url": "http://quick:8013", "served_by": "atlas-worker-fast-1", "backend": "llamacpp"},
    }


@pytest.fixture
def configured_routes(monkeypatch: pytest.MonkeyPatch, route_table: Dict[str, Dict[str, str]]) -> None:
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(route_table))
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    _load_route_targets.cache_clear()


def test_resolve_openai_route_uses_lane_key(configured_routes: None) -> None:
    route_key, target, upstream_model, error = openai_passthrough.resolve_openai_route("quick")
    assert error is None
    assert route_key == "quick"
    assert target is not None
    assert upstream_model == "quick"


def test_resolve_openai_route_missing_returns_error(
    monkeypatch: pytest.MonkeyPatch, route_table: Dict[str, Dict[str, str]]
) -> None:
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(route_table))
    monkeypatch.setattr(settings, "llm_route_default", "missing-default")
    _load_route_targets.cache_clear()

    _, _, _, error = openai_passthrough.resolve_openai_route("unknown-lane")
    assert error is not None
    assert error["error"]["type"] == "route_not_configured"


class TestOpenAIPassthroughHTTP:
    @pytest.fixture
    def client(self, configured_routes: None) -> TestClient:
        return TestClient(app)

    def test_post_chat_completions_missing_route(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "does-not-exist",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "route_not_configured"

    def test_post_chat_completions_proxies_to_upstream(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
            },
        )

        async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            assert url == "http://quick:8013/v1/chat/completions"
            assert kwargs["json"]["model"] == "quick"
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "quick",
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 8,
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "OK"

    def test_post_embeddings_requires_vector_host_url(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "orion_vector_host_url", None)
        response = client.post(
            "/v1/embeddings",
            json={"model": "orion-vector-host", "input": "hello"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "embeddings_not_configured"

    def test_post_embeddings_vector_host_shape(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "orion_vector_host_url", "http://vector-host:8320")

        mock_response = httpx.Response(
            200,
            json={
                "doc_id": "abc",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "embedding_dim": 1024,
            },
        )

        async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            assert url == "http://vector-host:8320/embedding"
            assert kwargs["json"]["text"] == "hello"
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        response = client.post(
            "/v1/embeddings",
            json={"model": "orion-vector-host", "input": "hello"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert body["model"] == "BAAI/bge-large-en-v1.5"
