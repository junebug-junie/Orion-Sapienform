"""Regression: fcc_motor must recover the real backend model from stream-json
"assistant" events, not just echo back the requested ~/.fcc/.env route alias.

Confirmed live 2026-08-19: MODEL_SONNET and MODEL_OPUS in ~/.fcc/.env both
route to the identical `llamacpp/chat` target, so `fcc_model_label` alone
cannot distinguish which real backend served a given turn. Also confirmed
live: llama.cpp's own Anthropic-compat `/v1/messages` endpoint echoes the
real served weights file (e.g. "/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf") in
the response's top-level "model" key regardless of the alias requested, and
orion-llm-gateway's anthropic_passthrough is a raw byte passthrough that
never rewrites that field -- so it should reach the CLI's own stream-json
"assistant" event under "message.model", given
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 is already set for both FCC
subprocess launch sites. The raw value is a full server-side filesystem
path, so the extraction reduces it to a basename with any weights-file
extension stripped before it can reach response_identity, a user-facing
"who answered" field, not an infra debug surface.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.harness.fcc_motor import (
    _route_key_from_fcc_env_value,
    _served_model_from_assistant,
    probe_current_served_model,
)


def test_served_model_extracted_and_reduced_to_basename() -> None:
    event = {
        "type": "assistant",
        "message": {
            "model": "/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf",
            "content": [{"type": "text", "text": "hi"}],
        },
    }
    assert _served_model_from_assistant(event) == "Qwen_Qwen3-8B-Q4_K_M"


def test_served_model_strips_whitespace() -> None:
    event = {"type": "assistant", "message": {"model": "  Qwen_Qwen3-8B-Q4_K_M.gguf  "}}
    assert _served_model_from_assistant(event) == "Qwen_Qwen3-8B-Q4_K_M"


def test_served_model_without_path_or_extension_passes_through() -> None:
    event = {"type": "assistant", "message": {"model": "qwen-36"}}
    assert _served_model_from_assistant(event) == "qwen-36"


def test_served_model_none_when_message_missing() -> None:
    assert _served_model_from_assistant({"type": "assistant"}) is None


def test_served_model_none_when_message_not_a_dict() -> None:
    assert _served_model_from_assistant({"type": "assistant", "message": "oops"}) is None


def test_served_model_none_when_model_field_missing() -> None:
    event = {"type": "assistant", "message": {"content": []}}
    assert _served_model_from_assistant(event) is None


def test_served_model_none_when_model_field_blank() -> None:
    event = {"type": "assistant", "message": {"model": "   "}}
    assert _served_model_from_assistant(event) is None


# --- _route_key_from_fcc_env_value ---------------------------------------


def test_route_key_splits_backend_and_route() -> None:
    assert _route_key_from_fcc_env_value("llamacpp/chat") == ("llamacpp", "chat")


def test_route_key_normalizes_backend_underscore() -> None:
    assert _route_key_from_fcc_env_value("llama_cpp/agent") == ("llama-cpp", "agent")


def test_route_key_none_when_no_separator() -> None:
    assert _route_key_from_fcc_env_value("qwen-36") is None


def test_route_key_none_when_blank() -> None:
    assert _route_key_from_fcc_env_value("") is None
    assert _route_key_from_fcc_env_value("/chat") is None
    assert _route_key_from_fcc_env_value("llamacpp/") is None


# --- probe_current_served_model -------------------------------------------


class _FakeRoutesResponse:
    def __init__(self, *, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _mock_client_returning(payload: dict, *, status_code: int = 200) -> MagicMock:
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    mock_client.get = AsyncMock(return_value=_FakeRoutesResponse(status_code=status_code, payload=payload))
    mock_client_cls = MagicMock(return_value=mock_client)
    return mock_client_cls


@pytest.mark.asyncio
async def test_probe_current_served_model_reads_cached_route_model() -> None:
    routes_payload = {
        "routes": [
            {"id": "chat", "model": "/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf", "status": "up"},
            {"id": "quick", "model": "/models/gguf/other.gguf", "status": "up"},
        ]
    }
    with patch(
        "orion.harness.fcc_motor.httpx.AsyncClient", _mock_client_returning(routes_payload)
    ):
        result = await probe_current_served_model(
            "MODEL_SONNET",
            env={"MODEL_SONNET": "llamacpp/chat"},
            gateway_url="http://llm-gateway:8210",
        )
    assert result == "Qwen_Qwen3-8B-Q4_K_M"


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_no_label() -> None:
    assert await probe_current_served_model(None, env={}) is None
    assert await probe_current_served_model("", env={}) is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_label_missing_from_env() -> None:
    assert await probe_current_served_model("MODEL_SONNET", env={}) is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_for_non_llamacpp_backend() -> None:
    """MODEL_HAIKU-style entries (e.g. nvidia_nim/z-ai/glm-5.2) aren't in
    orion-llm-gateway's route table -- must fail open without even calling
    out, not raise or misreport."""
    with patch("orion.harness.fcc_motor.httpx.AsyncClient") as mock_cls:
        result = await probe_current_served_model(
            "MODEL_HAIKU", env={"MODEL_HAIKU": "nvidia_nim/z-ai/glm-5.2"}
        )
    assert result is None
    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_route_not_found() -> None:
    routes_payload = {"routes": [{"id": "quick", "model": "x", "status": "up"}]}
    with patch(
        "orion.harness.fcc_motor.httpx.AsyncClient", _mock_client_returning(routes_payload)
    ):
        result = await probe_current_served_model(
            "MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/chat"}
        )
    assert result is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_route_model_not_yet_probed() -> None:
    """A down worker's route entry has model=None -- must not crash or
    return a placeholder."""
    routes_payload = {"routes": [{"id": "chat", "model": None, "status": "down"}]}
    with patch(
        "orion.harness.fcc_motor.httpx.AsyncClient", _mock_client_returning(routes_payload)
    ):
        result = await probe_current_served_model(
            "MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/chat"}
        )
    assert result is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_on_gateway_error_status() -> None:
    with patch(
        "orion.harness.fcc_motor.httpx.AsyncClient",
        _mock_client_returning({}, status_code=503),
    ):
        result = await probe_current_served_model(
            "MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/chat"}
        )
    assert result is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_on_transport_exception() -> None:
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    mock_client.get = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("orion.harness.fcc_motor.httpx.AsyncClient", MagicMock(return_value=mock_client)):
        result = await probe_current_served_model(
            "MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/chat"}
        )
    assert result is None
