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

from orion.harness.fcc_motor import _served_model_from_assistant


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
