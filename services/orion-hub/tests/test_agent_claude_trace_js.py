"""Gate: agent-claude live trace panel summarizes tool/file steps."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACE_JS = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "agent-claude-trace.js"


def test_agent_claude_trace_summarizes_tool_use_and_results() -> None:
    source = TRACE_JS.read_text(encoding="utf-8")
    assert "function formatToolInput(name, input)" in source
    assert "function summarizeContentBlocks(content)" in source
    assert "tool result" in source
    assert "OrionClaudeTrace = { appendLiveClaudeStep, summarizeStep }" in source
    assert "basename(path)" in source
    assert "Grep" in source
    assert "Bash" in source
