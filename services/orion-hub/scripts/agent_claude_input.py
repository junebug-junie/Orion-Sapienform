"""Prepare Hub agent-claude turn input. v2 adds slash-command dispatch."""
from __future__ import annotations

from dataclasses import dataclass

# Operational contract for local llamacpp (~64k ctx). Not conversational stance routing.
AGENT_CLAUDE_OPERATOR_BRIEF = """\
Hub agent-claude runs against a local gateway with a finite context window (~64k tokens).
Before Read on any file: prefer rg/Grep with a path or pattern. For large files (e.g. orion/bus/channels.yaml), use Read offset/limit in chunks — never load whole contract YAML in one read.
For GitHub PRs, issues, and repo metadata: use GitHub MCP tools (mcp__github). The gh CLI is not installed in the Hub container.
"""


@dataclass(frozen=True)
class TurnRequest:
    prompt: str


def prepare_agent_claude_input(text: str) -> TurnRequest:
    user_text = str(text or "").strip()
    if not user_text:
        return TurnRequest(prompt="")
    return TurnRequest(
        prompt=f"{AGENT_CLAUDE_OPERATOR_BRIEF.strip()}\n\nOperator request:\n{user_text}"
    )
