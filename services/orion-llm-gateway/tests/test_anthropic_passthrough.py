from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app import anthropic_passthrough
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
        "agent": {
            "url": "http://agent:8011",
            "served_by": "atlas-worker-1",
            "backend": "llamacpp",
            "model": "qwen-coder-local",
        },
        "quick": {"url": "http://quick:8013", "served_by": "atlas-worker-fast-1", "backend": "llamacpp"},
        "metacog": {"url": "http://metacog:8012", "served_by": "atlas-worker-2", "backend": "llamacpp"},
        "ollama_lane": {"url": "http://ollama:11434", "served_by": "ollama-host", "backend": "ollama"},
    }


@pytest.fixture
def configured_routes(monkeypatch: pytest.MonkeyPatch, route_table: Dict[str, Dict[str, str]]) -> None:
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(route_table))
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    _load_route_targets.cache_clear()


def test_normalize_anthropic_model_name() -> None:
    assert anthropic_passthrough.normalize_anthropic_model_name("llamacpp/agent") == "agent"
    assert anthropic_passthrough.normalize_anthropic_model_name("agent") == "agent"
    assert anthropic_passthrough.normalize_anthropic_model_name("quick") == "quick"


def test_resolve_anthropic_route_uses_upstream_model_alias(configured_routes: None) -> None:
    route_key, target, upstream_model, error = anthropic_passthrough.resolve_anthropic_route("llamacpp/agent")
    assert error is None
    assert route_key == "agent"
    assert target is not None
    assert upstream_model == "qwen-coder-local"


def test_resolve_anthropic_route_missing_returns_error(
    monkeypatch: pytest.MonkeyPatch, route_table: Dict[str, Dict[str, str]]
) -> None:
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(route_table))
    monkeypatch.setattr(settings, "llm_route_default", "missing-default")
    _load_route_targets.cache_clear()

    _, _, _, error = anthropic_passthrough.resolve_anthropic_route("unknown-lane")
    assert error is not None
    assert error["error"]["type"] == "route_not_configured"
    assert "unknown-lane" in error["error"]["message"]
    assert "agent" in error["error"]["available_routes"]


def test_resolve_anthropic_route_rejects_incompatible_backend(configured_routes: None) -> None:
    _, _, _, error = anthropic_passthrough.resolve_anthropic_route("ollama_lane")
    assert error is not None
    assert error["error"]["type"] == "backend_incompatible"
    assert error["error"]["route"] == "ollama_lane"


def test_resolve_anthropic_route_falls_back_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, route_table: Dict[str, Dict[str, str]]
) -> None:
    monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(route_table))
    monkeypatch.setattr(settings, "llm_route_default", "chat")
    _load_route_targets.cache_clear()

    route_key, target, upstream_model, error = anthropic_passthrough.resolve_anthropic_route(None)
    assert error is None
    assert route_key == "chat"
    assert upstream_model == "chat"


def test_build_models_list_payload(configured_routes: None) -> None:
    payload = anthropic_passthrough.build_models_list_payload()
    ids = [entry["id"] for entry in payload["data"]]
    assert ids == ["agent", "chat", "metacog", "quick"]
    assert "ollama_lane" not in ids



class TestAnthropicPassthroughHTTP:
    @pytest.fixture
    def client(self, configured_routes: None) -> TestClient:
        return TestClient(app)

    def test_get_v1_models_returns_route_keys(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        ids = [entry["id"] for entry in data["data"]]
        assert "agent" in ids
        assert "quick" in ids

    def test_post_v1_messages_missing_route(self, client: TestClient) -> None:
        response = client.post(
            "/v1/messages",
            json={
                "model": "does-not-exist",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["type"] == "route_not_configured"

    def test_post_v1_messages_unsupported_backend(self, client: TestClient) -> None:
        response = client.post(
            "/v1/messages",
            json={
                "model": "ollama_lane",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "backend_incompatible"

    @patch("app.anthropic_passthrough.httpx.AsyncClient")
    def test_post_v1_messages_non_streaming_proxy(self, mock_client_cls: MagicMock, client: TestClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"id":"msg_1","content":[{"type":"text","text":"OK"}]}'
        mock_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01", "x-api-key": "freecc"},
            json={
                "model": "llamacpp/agent",
                "max_tokens": 64,
                "stream": False,
                "messages": [{"role": "user", "content": "Say OK."}],
            },
        )

        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "OK"
        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.await_args.kwargs
        assert call_kwargs["json"]["model"] == "qwen-coder-local"
        assert mock_client.post.await_args.args[0] == "http://agent:8011/v1/messages"

    @patch("app.anthropic_passthrough.httpx.AsyncClient")
    def test_post_v1_messages_streaming_uses_streaming_response(
        self, mock_client_cls: MagicMock, client: TestClient
    ) -> None:
        class _Upstream:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                yield b"event: message_start\n\n"
                yield b"event: content_block_delta\n\n"

            async def aread(self) -> bytes:
                return b""

            async def aclose(self) -> None:
                return None

        mock_upstream = _Upstream()
        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_upstream)
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        with client.stream(
            "POST",
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": "agent",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "Say OK."}],
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            chunks: List[bytes] = list(response.iter_bytes())
            assert b"event: message_start" in b"".join(chunks)

        mock_client_cls.assert_called_once()
        mock_client.send.assert_awaited_once()
        assert mock_client.send.await_args.kwargs.get("stream") is True

    @patch("app.anthropic_passthrough.httpx.AsyncClient")
    def test_post_v1_messages_streaming_upstream_error_status(
        self, mock_client_cls: MagicMock, client: TestClient
    ) -> None:
        class _Upstream:
            status_code = 503
            headers = {"content-type": "application/json"}

            async def aread(self) -> bytes:
                return b'{"error":"upstream down"}'

            async def aclose(self) -> None:
                return None

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=_Upstream())
        mock_client.aclose = AsyncMock()
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/v1/messages",
            json={
                "model": "agent",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 503

    def test_get_head_and_options_messages(self, client: TestClient) -> None:
        get_resp = client.get("/v1/messages")
        assert get_resp.status_code == 200
        head = client.head("/v1/messages")
        assert head.status_code == 200
        options = client.options("/v1/messages")
        assert options.status_code == 204
        assert "POST" in options.headers.get("allow", "").upper() or "POST" in options.headers.get(
            "Access-Control-Allow-Methods", ""
        )

    def test_route_table_model_id_alias(
        self, monkeypatch: pytest.MonkeyPatch, route_table: Dict[str, Dict[str, str]]
    ) -> None:
        table = dict(route_table)
        table["quick"] = {
            **table["quick"],
            "model_id": "fast-local",
        }
        monkeypatch.setattr(settings, "llm_route_table_json", json.dumps(table))
        _load_route_targets.cache_clear()
        _, _, upstream_model, error = anthropic_passthrough.resolve_anthropic_route("quick")
        assert error is None
        assert upstream_model == "fast-local"
