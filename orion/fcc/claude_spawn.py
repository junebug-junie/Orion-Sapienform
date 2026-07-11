"""Shared ``claude -p`` argv helpers for FCC harness bridges."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Mapping, Sequence


def mcp_allowed_tool_patterns(mcp_servers: Mapping[str, Any]) -> List[str]:
    """Per-server allow patterns for Claude Code 2.1+ MCP pre-approval.

    Use ``mcp__<server>`` (not ``mcp__<server>__*`` or bare ``mcp__*``).
    See Claude Code IAM docs + anthropics/claude-code#5004.
    """
    return [f"mcp__{name}" for name in mcp_servers]


def mcp_disallowed_tool_patterns(mcp_servers: Mapping[str, Any]) -> List[str]:
    """Block Bash fallbacks that fail in headless Hub (gh not installed)."""
    blocked: List[str] = []
    if "github" in mcp_servers:
        blocked.append("Bash(gh *)")
    return blocked


def extend_mcp_argv(
    argv: List[str],
    mcp_config_path: Path,
    *,
    extra_allowed_tools: Sequence[str] | None = None,
) -> None:
    data = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    servers = data.get("mcpServers") or {}
    patterns = mcp_allowed_tool_patterns(servers)
    extra = list(extra_allowed_tools or [])
    argv.extend(["--mcp-config", str(mcp_config_path)])
    if patterns or extra:
        argv.append("--allowedTools")
        argv.extend(patterns)
        argv.extend(extra)
    disallowed = mcp_disallowed_tool_patterns(servers)
    if disallowed:
        argv.append("--disallowedTools")
        argv.extend(disallowed)


def claude_permission_argv(*, auto_approve: bool) -> List[str]:
    """Auto-approve tool permissions for non-interactive FCC turns."""
    if not auto_approve:
        return []
    # Claude Code 2.1+ rejects --dangerously-skip-permissions under root.
    if os.geteuid() == 0:
        return ["--permission-mode", "dontAsk"]
    return ["--dangerously-skip-permissions"]


def auto_approve_from_env(env_key: str | None = None) -> bool:
    """Whether to auto-approve when env is unset: root containers yes, host non-root yes."""
    if env_key:
        raw = os.environ.get(env_key, "").strip().lower()
        if raw in {"0", "false", "no", "off"}:
            return False
        if raw in {"1", "true", "yes", "on"}:
            return True
    if os.geteuid() == 0:
        return True
    return True
