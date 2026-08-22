"""Unit tests for _should_request_chat_reply_logprobs (CORTEX_CHAT_RETURN_LOGPROBS gate).

Regression coverage for a 2026-08-20 code review finding: an earlier version of this gate
keyed off llm_route == "chat", which is shared by stance_react (the real live reply) AND
harness_finalize_reflect / orion_voice_finalize (internal calls whose "chat" llm_route is
vestigial -- see _default_llm_route_for_step's docstring). That version would have silently
attached return_logprobs to those two internal calls too. This gate is keyed on exact
(verb_name, step_name) identity instead, via _REAL_CHAT_REPLY_STEPS.
"""
from types import SimpleNamespace
from unittest.mock import patch

from app.executor import _should_request_chat_reply_logprobs


def _step(verb_name: str, step_name: str) -> SimpleNamespace:
    return SimpleNamespace(verb_name=verb_name, step_name=step_name)


def test_stance_react_real_reply_step_requests_logprobs_when_enabled() -> None:
    """stance_react/llm_stance_react is the real, live chat-reply mechanism in this
    deployment (live-confirmed 2026-08-19/20: its content becomes final_text_assembly's
    source) -- it must get return_logprobs when the flag is on."""
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("stance_react", "llm_stance_react"), {}
        ) is True


def test_chat_general_final_step_requests_logprobs_when_enabled() -> None:
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("chat_general", "llm_chat_general"), {}
        ) is True


def test_disabled_by_default() -> None:
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=False)):
        assert _should_request_chat_reply_logprobs(
            _step("stance_react", "llm_stance_react"), {}
        ) is False


def test_harness_finalize_reflect_never_requests_logprobs_even_when_enabled() -> None:
    """The bug the review caught: harness_finalize_reflect shares llm_route == "chat"
    with stance_react, but is an internal reflection call, not Juniper's visible reply."""
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("harness_finalize_reflect", "llm_harness_finalize_reflect"), {}
        ) is False


def test_orion_voice_finalize_never_requests_logprobs_even_when_enabled() -> None:
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("orion_voice_finalize", "llm_orion_voice_finalize"), {}
        ) is False


def test_skips_when_response_format_set() -> None:
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("chat_general", "llm_chat_general"),
            {"response_format": {"type": "json_object"}},
        ) is False


def test_skips_when_return_json_set_without_response_format() -> None:
    """The bug the review caught: return_json alone (no response_format) still forces
    JSON-constrained decoding downstream (llm_backend.py:1116-1117 builds response_format
    from return_json when response_format itself is unset) -- a response_format-only check
    misses this."""
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("chat_general", "llm_chat_general"),
            {"return_json": True},
        ) is False


def test_skips_when_structured_output_schema_set() -> None:
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("chat_general", "llm_chat_general"),
            {"structured_output_schema": {"type": "object"}},
        ) is False


def test_skips_when_structured_output_method_set() -> None:
    with patch("app.executor.settings", SimpleNamespace(cortex_chat_return_logprobs=True)):
        assert _should_request_chat_reply_logprobs(
            _step("chat_general", "llm_chat_general"),
            {"structured_output_method": "json_schema"},
        ) is False
