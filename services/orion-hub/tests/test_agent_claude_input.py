from __future__ import annotations

from scripts.agent_claude_input import AGENT_CLAUDE_OPERATOR_BRIEF, TurnRequest, prepare_agent_claude_input


def test_operator_brief_prepended_to_prompt() -> None:
    result = prepare_agent_claude_input("  explain websocket_handler  ")
    assert isinstance(result, TurnRequest)
    assert "explain websocket_handler" in result.prompt
    assert AGENT_CLAUDE_OPERATOR_BRIEF.strip() in result.prompt
    assert "Operator request:" in result.prompt


def test_v1_empty_becomes_empty_string() -> None:
    result = prepare_agent_claude_input("   ")
    assert result.prompt == ""
