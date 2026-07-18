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


def setting_sources_argv(env_key: str) -> List[str]:
    """--setting-sources for FCC's claude subprocess: skip the repo's
    project-level CLAUDE.md/settings.json/hooks by default.

    Orion's FCC turns don't need the repo's own AGENTS.md development
    contract (written for a coding agent editing this repo, not a headless
    cognition turn) or the project-level hooks that come with it -- and
    dropping them isn't a safety regression: both orion-hub and
    orion-harness-governor bind-mount the repo read-only, so the
    destructive_git_guard hook this also drops has nothing left to protect
    that the read-only mount doesn't already block. MCP tool access is
    unaffected either way -- it's pre-approved via explicit
    --mcp-config/--allowedTools argv (extend_mcp_argv), never through
    settings.json, so --setting-sources has no bearing on it.

    Confirmed live: a `claude -p` call from a directory with a marker
    CLAUDE.md echoed the marker with default sources, returned nothing
    with `--setting-sources user,local`.

    Set the env key to empty to fall back to Claude Code's normal default
    (all three scopes: user, project, local).
    """
    raw = (os.environ[env_key] if env_key in os.environ else "user,local").strip()
    if not raw:
        return []
    return ["--setting-sources", raw]
