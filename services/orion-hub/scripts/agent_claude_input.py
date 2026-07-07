"""Prepare Hub agent-claude turn input. v2 adds slash-command dispatch."""
from __future__ import annotations

import os
from dataclasses import dataclass

from orion.fcc.github_repo_context import github_mcp_repo_brief_line

# Operational contract for local llamacpp (~64k ctx). Not conversational stance routing.
AGENT_CLAUDE_OPERATOR_BRIEF = """\
Hub agent-claude runs against a local gateway with a finite context window (~64k tokens).
Before Read on any file: prefer rg/Grep with a path or pattern. For large files (e.g. orion/bus/channels.yaml), use Read offset/limit in chunks — never load whole contract YAML in one read.
For GitHub PRs, issues, and repo metadata: use GitHub MCP tools (mcp__github). The gh CLI is not installed in the Hub container.
"""


def _operator_brief_for_workspace(workspace: str | None = None) -> str:
    brief = AGENT_CLAUDE_OPERATOR_BRIEF.strip()
    ws = workspace or os.environ.get("HUB_AGENT_CLAUDE_WORKSPACE")
    github_line = github_mcp_repo_brief_line(workspace=ws)
    if github_line:
        brief = f"{brief}\n{github_line}"
    return brief


@dataclass(frozen=True)
class TurnRequest:
    prompt: str


def prepare_agent_claude_input(text: str) -> TurnRequest:
    user_text = str(text or "").strip()
    if not user_text:
        return TurnRequest(prompt="")
    return TurnRequest(
        prompt=f"{_operator_brief_for_workspace()}\n\nOperator request:\n{user_text}"
    )
