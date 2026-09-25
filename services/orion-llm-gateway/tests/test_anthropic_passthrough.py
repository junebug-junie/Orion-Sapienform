from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import anthropic_passthrough
from app.main import app
from app.settings import settings


@pytest.fixture
def configured_routes(monkeypatch: pytest.MonkeyPatch, fake_pool) -> Any:
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    return fake_pool


def test_normalize_anthropic_model_name() -> None:
    assert anthropic_passthrough.normalize_anthropic_model_name("llamacpp/agent") == "agent"
    assert anthropic_passthrough.normalize_anthropic_model_name("agent") == "agent"
    assert anthropic_passthrough.normalize_anthropic_model_name("quick") == "quick"
    assert anthropic_passthrough.normalize_anthropic_model_name("llamacpp/harness") == "harness"


@pytest.mark.parametrize("system", [None, "Original instructions", [
    {"type": "text", "text": "Original instructions", "cache_control": {"type": "ephemeral"}}
]])
def test_hook_system_context_preserves_blocks_and_tool_order(system: Any) -> None:
    hook = {"type": "text", "text": "SessionStart hook additional context", "cache_control": {"type": "ephemeral"}}
    conversation = [
        {"role": "user", "content": "Inspect evidence"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "Evidence"}]},
    ]
    body = {"system": system, "messages": [conversation[0], {"role": "system", "content": [hook]}, *conversation[1:], {"role": "system", "content": "Later context"}]}
    original = json.loads(json.dumps(body))
    normalized = anthropic_passthrough.normalize_anthropic_system_messages(body)
    assert normalized["messages"] == conversation
    assert normalized["system"][-3:] == [hook, {"type": "text", "text": "\n\n"}, {"type": "text", "text": "Later context"}]
    if system:
        expected = [{"type": "text", "text": system}] if isinstance(system, str) else system
        assert normalized["system"][:len(expected)] == expected
        assert normalized["system"][len(expected)] == {"type": "text", "text": "\n\n"}
    assert body == original


def test_standard_anthropic_body_unchanged() -> None:
    body = {"system": "Instructions", "messages": [{"role": "user", "content": "Hi"}]}
    assert anthropic_passthrough.normalize_anthropic_system_messages(body) == body


def test_resolve_anthropic_route_resolves_harness(configured_routes: Any) -> None:
    # This is the literal live path: ~/.fcc/.env sets MODEL=llamacpp/harness, and this is what
    # a real Claude Code CLI turn's `model` field resolves to via this function.
    route_key, upstream_model, error = anthropic_passthrough.resolve_anthropic_route("llamacpp/harness")
    assert error is None
    assert route_key == "harness"
    assert upstream_model == "harness"


def test_resolve_anthropic_route_missing_returns_error(
    monkeypatch: pytest.MonkeyPatch, configured_routes: Any
) -> None:
    monkeypatch.setattr(settings, "llm_route_default", "missing-default")
    _, _, error = anthropic_passthrough.resolve_anthropic_route("unknown-lane")
    assert error is not None
    assert error["error"]["type"] == "route_not_in_gpu_pool"
    assert "unknown-lane" in error["error"]["message"]
    assert "agent" in error["error"]["available_routes"]


def test_resolve_anthropic_route_falls_back_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, configured_routes: Any
) -> None:
    monkeypatch.setattr(settings, "llm_route_default", "quick")
    route_key, upstream_model, error = anthropic_passthrough.resolve_anthropic_route(None)
    assert error is None
    assert route_key == "quick"
    assert upstream_model == "quick"


def test_build_models_list_payload_lists_every_pool_route(configured_routes: Any) -> None:
    payload = anthropic_passthrough.build_models_list_payload()
    ids = [entry["id"] for entry in payload["data"]]
    assert ids == sorted(["chat", "harness", "agent", "metacog", "metacog_background", "quick",
                          "quick_background", "agent-burst", "chat-burst"])
    # Placement is per call, so no static served_by claim.
    assert all(entry["served_by"] is None for entry in payload["data"])


class _StreamUpstream:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, pool) -> None:
        self.pool = pool
        self.active_during_stream: List[int] = []

    async def aiter_bytes(self):
        self.active_during_stream.append(self.pool.active)
        yield b"event: message_start\n\n"
        self.active_during_stream.append(self.pool.active)
        yield b"event: content_block_delta\n\n"

    async def aread(self) -> bytes:
        return b""

    async def aclose(self) -> None:
        return None


def _stream_client(upstream: Any) -> MagicMock:
    mock_client = MagicMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=upstream)
    mock_client.aclose = AsyncMock()
    return mock_client


def _post_client(status: int, content: bytes) -> AsyncMock:
    mock_response = MagicMock()
    mock_response.status_code = status
    mock_response.content = content
    mock_response.headers = {"content-type": "application/json"}
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


class TestAnthropicPassthroughHTTP:
    @pytest.fixture
    def client(self, configured_routes: Any) -> TestClient:
        return TestClient(app)

    def test_get_v1_models_returns_route_keys(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        ids = [entry["id"] for entry in response.json()["data"]]
        assert "agent" in ids
        assert "quick" in ids

    def test_post_v1_messages_missing_route_takes_no_lease(self, client: TestClient, configured_routes: Any) -> None:
        response = client.post(
            "/v1/messages",
            json={"model": "does-not-exist", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "route_not_in_gpu_pool"
        assert configured_routes.calls == []

    @patch("app.passthrough_proxy.httpx.AsyncClient")
    def test_post_v1_messages_non_streaming_goes_to_granted_url(
        self, mock_client_cls: MagicMock, client: TestClient, configured_routes: Any
    ) -> None:
        mock_client = _post_client(200, b'{"id":"msg_1","content":[{"type":"text","text":"OK"}]}')
        mock_client_cls.return_value = mock_client

        response = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01", "x-api-key": "freecc"},
            json={
                "model": "llamacpp/agent",
                "max_tokens": 64,
                "stream": False,
                "system": "s" * 40,
                "messages": [{"role": "user", "content": "Say OK."},
                             {"role": "system", "content": "SessionStart context"}],
            },
        )

        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "OK"
        call_kwargs = mock_client.post.await_args.kwargs
        assert call_kwargs["json"]["model"] == "agent"
        assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "Say OK."}]
        assert mock_client.post.await_args.args[0] == "http://pool-agent:8015/v1/messages"
        pool = configured_routes
        assert len(pool.calls) == 1
        assert pool.calls[0]["work_class"] == "agent"
        assert pool.calls[0]["holder"] == "http:anthropic"
        # system (40 + hoisted 20 + separator 2 chars) + message 7 chars, /4, + max_tokens
        assert pool.calls[0]["min_ctx_tokens"] > 64
        assert pool.releases == ["ok"]

    @pytest.mark.parametrize("content", [1, {"text": "invalid block container"}])
    def test_invalid_hook_context_never_takes_a_lease(
        self, client: TestClient, content: Any, configured_routes: Any
    ) -> None:
        response = client.post("/v1/messages", json={
            "model": "agent", "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": content}],
        })
        assert response.status_code == 400
        assert configured_routes.calls == []

    @patch("app.passthrough_proxy.httpx.AsyncClient")
    def test_streaming_holds_the_lease_until_the_stream_ends(
        self, mock_client_cls: MagicMock, client: TestClient, configured_routes: Any
    ) -> None:
        pool = configured_routes
        upstream = _StreamUpstream(pool)
        mock_client = _stream_client(upstream)
        mock_client_cls.return_value = mock_client

        with client.stream(
            "POST", "/v1/messages", headers={"anthropic-version": "2023-06-01"},
            json={"model": "agent", "max_tokens": 64, "stream": True,
                  "messages": [{"role": "user", "content": "Say OK."}]},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            chunks: List[bytes] = list(response.iter_bytes())
            assert b"event: message_start" in b"".join(chunks)

        assert mock_client.send.await_args.kwargs.get("stream") is True
        assert mock_client.build_request.call_args.args[1] == "http://pool-agent:8015/v1/messages"
        assert upstream.active_during_stream == [1, 1]  # lease held while bytes flow
        assert pool.active == 0 and pool.releases == ["ok"]
        mock_client.aclose.assert_awaited()

    @patch("app.passthrough_proxy.httpx.AsyncClient")
    def test_streaming_upstream_error_status_releases_as_failure(
        self, mock_client_cls: MagicMock, client: TestClient, configured_routes: Any
    ) -> None:
        class _Upstream:
            status_code = 503
            headers = {"content-type": "application/json"}

            async def aread(self) -> bytes:
                return b'{"error":"upstream down"}'

            async def aclose(self) -> None:
                return None

        mock_client_cls.return_value = _stream_client(_Upstream())
        response = client.post(
            "/v1/messages",
            json={"model": "agent", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 503
        assert configured_routes.releases == ["upstream_error"]

    @patch("app.passthrough_proxy.httpx.AsyncClient")
    def test_context_overflow_re_leases_once_with_bigger_min_ctx(
        self, mock_client_cls: MagicMock, client: TestClient, configured_routes: Any
    ) -> None:
        pool = configured_routes
        pool.choose = lambda kw: "agent" if len(pool.calls) == 1 else "chat"
        overflow = _post_client(400, b'{"error":{"message":"the request exceeds the available context size"}}')
        ok = _post_client(200, b'{"id":"msg_2"}')
        mock_client_cls.side_effect = [overflow, ok]

        response = client.post("/v1/messages", json={
            "model": "agent", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})

        assert response.status_code == 200
        assert [c["min_ctx_tokens"] for c in pool.calls][1] == 32768 + 1
        assert pool.releases == ["upstream_error", "ok"]
        assert ok.post.await_args.args[0] == "http://pool-chat:8011/v1/messages"

    def test_pool_unavailable_is_a_typed_503(self, client: TestClient, configured_routes: Any) -> None:
        configured_routes.unavailable = "no_serviceable_role"
        response = client.post("/v1/messages", json={
            "model": "metacog", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 503
        err = response.json()["error"]
        assert err["type"] == "gpu_pool_unavailable"
        assert err["reason"] == "no_serviceable_role"
        assert err["work_class"] == "metacog"

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
