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


_CC = {"type": "ephemeral"}
_SYSTEM = [
    {"type": "text", "text": "x-anthropic-billing-header: cc_version=test;"},
    {"type": "text", "text": "You are a test agent.", "cache_control": _CC},
]
_REMINDERS = [
    "SessionStart hook additional context: <context>fixture</context>",
    "<total_tokens>14987040 tokens left</total_tokens>",
    "PreToolUse:Read hook additional context: <tip>fixture</tip>",
    "PreToolUse:Bash hook additional context: <tip>fixture</tip>",
]


def _claude_code_step(n: int) -> dict:
    """Claude Code 2.1.287 request shape for step n (1-based), per the captured probe.

    The reminder that is new on this step is a one-block list carrying
    cache_control; Claude Code resends earlier reminders as plain strings.
    """
    messages: list = [
        {"role": "user", "content": [{"type": "text", "text": "Inspect the evidence."}]},
    ]

    def reminder(i: int) -> dict:
        if i == n - 1:
            return {"role": "system", "content": [{"type": "text", "text": _REMINDERS[i], "cache_control": _CC}]}
        return {"role": "system", "content": _REMINDERS[i]}

    messages.append(reminder(0))
    for step in range(1, n):
        tool_id = f"toolu_{step}"
        messages.append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"step {step} plan", "signature": ""},
            {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"command": f"echo {step}"}},
        ]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": f"out {step}"}]})
        messages.append(reminder(step))
    return {"model": "llamacpp/harness", "system": json.loads(json.dumps(_SYSTEM)), "messages": messages,
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}]}


def _rendered_view(value: Any) -> Any:
    """What llama.cpp renders: cache_control is a caching hint, not prompt text."""
    if isinstance(value, dict):
        return {k: _rendered_view(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_rendered_view(v) for v in value]
    return value


def _joined_text(message: dict) -> str:
    return "".join(block["text"] for block in message["content"] if block.get("type") == "text")


def test_claude_code_steps_forward_append_only() -> None:
    forwarded = [anthropic_passthrough.normalize_anthropic_system_messages(_claude_code_step(n)) for n in range(1, 5)]
    for previous, current in zip(forwarded, forwarded[1:]):
        # (a) the system block no longer grows step to step
        assert current["system"] == previous["system"] == _SYSTEM
        # (b) step N's conversation is a strict prefix of step N+1's
        assert len(current["messages"]) > len(previous["messages"])
        assert _rendered_view(current["messages"][: len(previous["messages"])]) == _rendered_view(previous["messages"])
    for body in forwarded:
        assert all(message["role"] != "system" for message in body["messages"])


def test_mid_conversation_system_becomes_wrapped_user_turn_in_place() -> None:
    body = _claude_code_step(2)
    original = json.loads(json.dumps(body))
    forwarded = anthropic_passthrough.normalize_anthropic_system_messages(body)
    roles = [message["role"] for message in forwarded["messages"]]
    assert roles == ["user", "user", "assistant", "user", "user"]
    assert _joined_text(forwarded["messages"][1]) == f"<system-reminder>\n{_REMINDERS[0]}\n</system-reminder>"
    newest = forwarded["messages"][-1]
    assert newest["content"] == [{"type": "text", "text": f"<system-reminder>\n{_REMINDERS[1]}\n</system-reminder>", "cache_control": _CC}]
    assert body == original


def test_reminder_preserves_every_block_and_wraps_non_text_edges() -> None:
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
    body = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": [image, {"type": "text", "text": "a", "cache_control": _CC}, image]},
    ]}
    content = anthropic_passthrough.normalize_anthropic_system_messages(body)["messages"][1]["content"]
    assert content == [
        {"type": "text", "text": "<system-reminder>\n"}, image,
        {"type": "text", "text": "a", "cache_control": _CC}, image,
        {"type": "text", "text": "\n</system-reminder>"},
    ]


def test_reminder_between_tool_use_and_tool_result_waits_for_the_result() -> None:
    tool_use = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}
    tool_result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}
    body = {"messages": [{"role": "user", "content": "Go"}, tool_use, {"role": "system", "content": "hook"}, tool_result]}
    messages = anthropic_passthrough.normalize_anthropic_system_messages(body)["messages"]
    assert messages[:3] == [body["messages"][0], tool_use, tool_result]
    assert _joined_text(messages[3]) == "<system-reminder>\nhook\n</system-reminder>"
    # still forwarded if the request ends before the result arrives
    trailing = anthropic_passthrough.normalize_anthropic_system_messages({"messages": body["messages"][:3]})["messages"]
    assert [m["role"] for m in trailing] == ["user", "assistant", "user"]


def test_empty_late_system_message_is_dropped() -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}, {"role": "system", "content": ""}, {"role": "system", "content": []}]}
    assert anthropic_passthrough.normalize_anthropic_system_messages(body)["messages"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.parametrize("system", [None, "Original instructions", [
    {"type": "text", "text": "Original instructions", "cache_control": {"type": "ephemeral"}}
]])
def test_leading_system_messages_still_hoisted(system: Any) -> None:
    hook = {"type": "text", "text": "Leading hook context", "cache_control": _CC}
    conversation = [{"role": "user", "content": "Inspect evidence"}]
    body = {"system": system, "messages": [{"role": "system", "content": [hook]}, {"role": "system", "content": "Second"}, *conversation]}
    original = json.loads(json.dumps(body))
    normalized = anthropic_passthrough.normalize_anthropic_system_messages(body)
    assert normalized["messages"] == conversation
    assert normalized["system"][-3:] == [hook, {"type": "text", "text": "\n\n"}, {"type": "text", "text": "Second"}]
    if system:
        expected = [{"type": "text", "text": system}] if isinstance(system, str) else system
        assert normalized["system"][: len(expected)] == expected
        assert normalized["system"][len(expected)] == {"type": "text", "text": "\n\n"}
    assert body == original


def test_no_system_key_added_when_nothing_hoisted() -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}, {"role": "system", "content": "late"}]}
    normalized = anthropic_passthrough.normalize_anthropic_system_messages(body)
    assert "system" not in normalized
    assert [m["role"] for m in normalized["messages"]] == ["user", "user"]


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
        assert call_kwargs["json"]["system"] == [{"type": "text", "text": "s" * 40}]
        assert call_kwargs["json"]["messages"] == [
            {"role": "user", "content": "Say OK."},
            {"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nSessionStart context\n</system-reminder>"}]},
        ]
        assert mock_client.post.await_args.args[0] == "http://pool-agent:8015/v1/messages"
        pool = configured_routes
        assert len(pool.calls) == 1
        assert pool.calls[0]["work_class"] == "agent"
        assert pool.calls[0]["holder"] == "http:anthropic"
        # system 40 chars + messages (7 + wrapped reminder), /4, + max_tokens
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
        overflow = _post_client(400, b'{"error":{"message":"the request exceeds the available context size"}}')
        ok = _post_client(200, b'{"id":"msg_2"}')
        mock_client_cls.side_effect = [overflow, ok]

        # metacog (4096/slot) overflows -> the pool places ctx 4097 on the first bigger metacog-class
        # role, agent (131072/slot).
        response = client.post("/v1/messages", json={
            "model": "metacog", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})

        assert response.status_code == 200
        assert [c["min_ctx_tokens"] for c in pool.calls][1] == 4096 + 1
        assert pool.releases == ["ok", "ok"]  # an overflow is not the GPU's failure
        assert overflow.post.await_args.args[0] == "http://pool-metacog:8012/v1/messages"
        assert ok.post.await_args.args[0] == "http://pool-agent:8015/v1/messages"

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
