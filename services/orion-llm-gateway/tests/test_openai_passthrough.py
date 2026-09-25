from __future__ import annotations

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app import openai_passthrough
from app.main import app
from app.settings import settings


@pytest.fixture
def configured_routes(monkeypatch: pytest.MonkeyPatch, fake_pool) -> Any:
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    return fake_pool


def test_resolve_openai_route_uses_lane_key(configured_routes: Any) -> None:
    route_key, upstream_model, error = openai_passthrough.resolve_openai_route("quick")
    assert error is None
    assert route_key == "quick"
    assert upstream_model == "quick"


def test_resolve_openai_route_missing_returns_error(
    monkeypatch: pytest.MonkeyPatch, configured_routes: Any
) -> None:
    monkeypatch.setattr(settings, "llm_route_default", "missing-default")
    _, _, error = openai_passthrough.resolve_openai_route("unknown-lane")
    assert error is not None
    assert error["error"]["type"] == "route_not_in_gpu_pool"


class TestOpenAIPassthroughHTTP:
    @pytest.fixture
    def client(self, configured_routes: Any) -> TestClient:
        return TestClient(app)

    def test_post_chat_completions_missing_route(self, client: TestClient, configured_routes: Any) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "route_not_in_gpu_pool"
        assert configured_routes.calls == []

    def test_post_chat_completions_proxies_to_granted_url(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, configured_routes: Any
    ) -> None:
        pool = configured_routes
        seen: List[int] = []
        mock_response = httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
            },
        )

        async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            assert url == "http://pool-fast:8013/v1/chat/completions"
            assert kwargs["json"]["model"] == "quick"
            seen.append(pool.active)
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "quick", "messages": [{"role": "user", "content": "Say OK" * 4}], "max_tokens": 8},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "OK"
        assert seen == [1]
        assert pool.calls[0]["work_class"] == "fast"
        assert pool.calls[0]["holder"] == "http:openai"
        assert pool.calls[0]["min_ctx_tokens"] == 6 + 8  # ceil(24 chars / 4) + max_tokens
        assert pool.releases == ["ok"]

    def test_background_route_leases_with_background_priority(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, configured_routes: Any
    ) -> None:
        async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
        response = client.post("/v1/chat/completions", json={
            "model": "quick_background", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200
        call = configured_routes.calls[0]
        assert (call["work_class"], call["priority"]) == ("fast", "background")
        assert call["deadline_sec"] == settings.llm_gateway_pool_background_wait_sec

    def test_upstream_http_error_releases_as_failure(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, configured_routes: Any
    ) -> None:
        async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "boom"}})

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
        response = client.post("/v1/chat/completions", json={
            "model": "quick", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 500
        assert configured_routes.releases == ["upstream_error"]

    @patch("app.passthrough_proxy.httpx.AsyncClient")
    def test_streaming_holds_the_lease_until_the_stream_ends(
        self, mock_client_cls: MagicMock, client: TestClient, configured_routes: Any
    ) -> None:
        pool = configured_routes
        active: List[int] = []

        class _Upstream:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                active.append(pool.active)
                yield b"data: {}\n\n"
                active.append(pool.active)
                yield b"data: [DONE]\n\n"

            async def aclose(self) -> None:
                return None

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=_Upstream())
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with client.stream("POST", "/v1/chat/completions", json={
            "model": "quick", "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as response:
            assert response.status_code == 200
            body = b"".join(response.iter_bytes())
        assert b"[DONE]" in body
        assert active == [1, 1]
        assert pool.active == 0 and pool.releases == ["ok"]
        assert mock_client.build_request.call_args.args[1] == "http://pool-fast:8013/v1/chat/completions"

    @patch("app.passthrough_proxy.httpx.AsyncClient")
    def test_stream_that_breaks_mid_way_releases_as_failure(
        self, mock_client_cls: MagicMock, client: TestClient, configured_routes: Any
    ) -> None:
        class _Upstream:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                yield b"data: {}\n\n"
                raise httpx.ReadError("connection reset")

            async def aclose(self) -> None:
                return None

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=_Upstream())
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with pytest.raises(Exception):  # an ExceptionGroup from the ASGI task group
            with client.stream("POST", "/v1/chat/completions", json={
                "model": "quick", "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as response:
                b"".join(response.iter_bytes())
        assert configured_routes.active == 0
        assert configured_routes.releases == ["upstream_error"]

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
