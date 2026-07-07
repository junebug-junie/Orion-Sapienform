"""Shared ``claude -p`` argv helpers for FCC harness bridges."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Mapping


def mcp_allowed_tool_patterns(mcp_servers: Mapping[str, Any]) -> List[str]:
    """Per-server allow patterns for Claude Code 2.1+ (``mcp__*`` is rejected)."""
    return [f"mcp__{name}__*" for name in mcp_servers]


def extend_mcp_argv(argv: List[str], mcp_config_path: Path) -> None:
    data = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    patterns = mcp_allowed_tool_patterns(data.get("mcpServers") or {})
    argv.extend(["--mcp-config", str(mcp_config_path)])
    if patterns:
        argv.append("--allowedTools")
        argv.extend(patterns)


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
