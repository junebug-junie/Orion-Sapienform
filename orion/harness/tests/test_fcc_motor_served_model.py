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


# --- probe_current_served_model / probe_route_runtime (GPU pool state, stage 6.3) ---------


from orion.gpu_pool.config import load_pool_config  # noqa: E402
from orion.harness.fcc_motor import held_role_window, probe_route_runtime  # noqa: E402

_CFG_PAYLOAD = load_pool_config().model_dump(mode="json", by_alias=True, exclude={"digest"})


def _pool_state(*roles: dict, config: bool = True) -> dict:
    state = {"generated_at": "2026-09-30T00:00:00Z",
             "cards": [{"card": "gpu0", "vram_gb": 32, "lendable": True, "lent": False}],
             "roles": list(roles)}
    if config:
        state["config"] = _CFG_PAYLOAD
    return state


def _role(role: str, status: str = "confirmed", **kw) -> dict:
    return {"role": role, "kind": "llm", "cards": ["gpu0"], "url": "http://h:8011", "status": status,
            "model_file": f"{role}.gguf", "ctx_per_slot": 8192, **kw}


@pytest.mark.asyncio
async def test_probe_current_served_model_reads_the_route_model_from_pool_state() -> None:
    state = _pool_state(_role("chat", model_path="/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf"))
    result = await probe_current_served_model(
        "MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/chat"}, pool_state=state,
    )
    assert result == "Qwen_Qwen3-8B-Q4_K_M"


@pytest.mark.asyncio
async def test_probe_route_runtime_returns_the_landing_roles_window() -> None:
    # metacog's own role is down: the route lands on fast, so fast's model and window.
    state = _pool_state(_role("metacog", "down"), _role("fast", ctx_per_slot=4096))
    assert await probe_route_runtime("llamacpp/metacog", env={}, pool_state=state) == ("fast", 4096)


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_no_label() -> None:
    state = _pool_state(_role("chat"))
    assert await probe_current_served_model(None, env={}, pool_state=state) is None
    assert await probe_current_served_model("", env={}, pool_state=state) is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_label_missing_from_env() -> None:
    assert await probe_current_served_model("MODEL_SONNET", env={}, pool_state=_pool_state(_role("chat"))) is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_for_non_llamacpp_backend() -> None:
    """MODEL_HAIKU-style entries (e.g. nvidia_nim/z-ai/glm-5.2) are not pool routes -- must fail
    open, not raise or misreport."""
    result = await probe_current_served_model(
        "MODEL_HAIKU", env={"MODEL_HAIKU": "nvidia_nim/z-ai/glm-5.2"}, pool_state=_pool_state(_role("chat"))
    )
    assert result is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_route_not_in_pool_config() -> None:
    result = await probe_current_served_model(
        "MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/no-such-route"}, pool_state=_pool_state(_role("chat"))
    )
    assert result is None


@pytest.mark.asyncio
async def test_probe_current_served_model_none_when_worker_down() -> None:
    """A down route has no model and no window -- never a placeholder."""
    state = _pool_state(_role("chat", "down"))
    assert await probe_route_runtime("MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/chat"},
                                     pool_state=state) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [
    None,  # pool unreachable
    _pool_state(_role("chat"), config=False),  # a broadcast frame: cannot map route -> role
])
async def test_probe_is_unknown_without_usable_pool_state(state) -> None:
    """Pool unreachable is an honest unknown (None, None): never the route default guessed."""
    assert await probe_route_runtime("MODEL_SONNET", env={"MODEL_SONNET": "llamacpp/chat"},
                                     pool_state=state) == (None, None)


def test_probe_makes_no_http_call() -> None:
    """Stage 6.3 pin: the motor no longer reads the gateway's retiring GET /routes."""
    import inspect

    import orion.harness.fcc_motor as motor

    assert not hasattr(motor, "httpx")
    source = inspect.getsource(motor.probe_route_runtime)
    assert "AsyncClient" not in source and "urlopen" not in source and "gateway_url" not in source


def test_held_role_window_is_the_grants_ctx_not_the_route_default() -> None:
    state = _pool_state(_role("agent", ctx_per_slot=32768), _role("agent-gpu2", ctx_per_slot=65536))
    assert held_role_window(state, "agent-gpu2") == 65536
    assert held_role_window(state, "missing") is None
    assert held_role_window(_pool_state(_role("chat", "mismatch")), "chat") is None
    assert held_role_window(None, "chat") is None
