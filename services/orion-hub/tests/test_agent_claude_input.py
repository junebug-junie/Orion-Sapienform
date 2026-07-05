from __future__ import annotations

from scripts.agent_claude_input import TurnRequest, prepare_agent_claude_input


def test_v1_pass_through_prompt() -> None:
    result = prepare_agent_claude_input("  explain websocket_handler  ")
    assert isinstance(result, TurnRequest)
    assert result.prompt == "explain websocket_handler"


def test_v1_empty_becomes_empty_string() -> None:
    result = prepare_agent_claude_input("   ")
    assert result.prompt == ""
