from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm_backend import _maybe_publish_spark_introspect, _should_publish_spark_candidate
from app.models import ChatBody, ChatMessage


def _body(**kwargs) -> ChatBody:
    defaults = {
        "messages": [ChatMessage(role="user", content="hello")],
        "trace_id": "00000000-0000-0000-0000-00000000a001",
        "source": "cortex-exec",
    }
    defaults.update(kwargs)
    return ChatBody(**defaults)


def test_should_publish_on_normal_chat_completion() -> None:
    body = _body(
        verb="chat_general",
        options={"llm_route": "chat"},
    )
    meta = {"latest_user_message": "hello", "trace_verb": "chat_general"}
    assert _should_publish_spark_candidate(body, meta) is True


@pytest.mark.parametrize(
    "options,verb,meta_extra",
    [
        ({"purpose": "introspect"}, None, {}),
        ({"purpose": "classify"}, None, {}),
        ({"skip_spark_candidate_publish": True}, None, {}),
        ({}, "introspect_spark", {}),
        (
            {
                "post_turn": True,
                "skip_chat_stance_inputs": True,
            },
            None,
            {},
        ),
    ],
)
def test_should_not_publish_internal_rpc_completions(options, verb, meta_extra) -> None:
    body = _body(
        verb=verb,
        options=options,
        messages=[ChatMessage(role="user", content="Analyze the state shift.")],
    )
    meta = {
        "latest_user_message": "Analyze the state shift.",
        "trace_verb": verb or options.get("purpose") or "unknown",
        **meta_extra,
    }
    assert _should_publish_spark_candidate(body, meta) is False


def test_maybe_publish_skips_introspect_spark(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[dict] = []

    async def _capture(payload: dict) -> None:
        published.append(payload)

    monkeypatch.setattr("app.llm_backend._run_async", lambda coro: None)
    with patch("app.llm_backend._publish_spark_introspect", side_effect=_capture):
        body = _body(
            verb="introspect_spark",
            options={
                "purpose": "introspect",
                "post_turn": True,
                "skip_chat_stance_inputs": True,
            },
            messages=[ChatMessage(role="user", content="Analyze the state shift.")],
        )
        meta = {
            "latest_user_message": "Analyze the state shift.",
            "trace_verb": "introspect_spark",
        }
        _maybe_publish_spark_introspect(body, meta, "internal introspection text")
    assert published == []


def test_maybe_publish_allows_hub_like_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def _run_async(coro):
        calls.append(coro)

    monkeypatch.setattr("app.llm_backend._run_async", _run_async)
    body = _body(verb="chat_general", source="cortex-exec")
    meta = {"latest_user_message": "hello", "trace_verb": "chat_general"}
    _maybe_publish_spark_introspect(body, meta, "hi there")
    assert len(calls) == 1
