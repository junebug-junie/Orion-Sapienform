"""Tests for join_openai_message_content helper."""

from __future__ import annotations

from orion.llm.openai_message_content import join_openai_message_content


def test_join_string_content() -> None:
    assert join_openai_message_content('{"ok": true}') == '{"ok": true}'


def test_join_list_content_skips_reasoning_parts() -> None:
    content = [
        {"type": "reasoning", "text": "hidden"},
        {"type": "text", "text": '{"ok": true}'},
    ]
    assert join_openai_message_content(content) == '{"ok": true}'
