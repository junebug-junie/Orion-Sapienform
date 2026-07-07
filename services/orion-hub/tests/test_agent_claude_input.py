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


def test_prepare_agent_claude_input_includes_github_brief_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    result = prepare_agent_claude_input("latest pr title")
    assert "list_pull_requests" in result.prompt
    assert "perPage=1" in result.prompt
    assert "owner='junebug-junie'" in result.prompt
