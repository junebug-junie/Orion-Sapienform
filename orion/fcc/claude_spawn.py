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


def extend_mcp_argv(
    argv: List[str],
    mcp_config_path: Path,
    *,
    extra_allowed_tools: Sequence[str] | None = None,
) -> None:
    """Emit ``--mcp-config`` + per-server ``--allowedTools`` for a FCC turn.

    Deliberately emits NO ``--disallowedTools``. Until 2026-08-13 this
    appended ``Bash(gh *)`` whenever the github MCP server was present, on
    the premise that ``gh`` was not installed in the headless Hub container
    so the model would waste turns on a CLI fallback that could not work.
    That premise is dead: ``gh`` 2.63.2 is bind-mounted into BOTH containers
    that spawn ``claude -p`` (``/home/athena/.local/bin/gh`` ->
    ``/usr/local/bin/gh``) and authenticated as ``junebug-junie`` with
    ``repo`` scope via ``/root/.config/gh``.

    Keeping the deny actively broke Orion's only route to opening a PR, and
    a deny rule beats ``--permission-mode bypassPermissions`` -- so ``gh``
    was the single tool Orion could not run while holding otherwise
    unrestricted Bash. The github MCP server is rendered read-only
    (``GITHUB_READ_ONLY=1``, see orion/fcc/mcp_config.py), so it exposes no
    ``create_pull_request``; with ``Bash(gh *)`` denied as well, BOTH
    PR-creation paths were closed. Confirmed live 2026-08-13 from the
    running governor's own rendered argv.

    Do not re-add a ``gh`` deny to steer tool choice. If ``gh`` should ever
    be blocked again, verify first that it is actually absent or unusable
    in every container listed above, and say so here.
    """
    data = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    servers = data.get("mcpServers") or {}
    patterns = mcp_allowed_tool_patterns(servers)
    extra = list(extra_allowed_tools or [])
    argv.extend(["--mcp-config", str(mcp_config_path)])
    if patterns or extra:
        argv.append("--allowedTools")
        argv.extend(patterns)
        argv.extend(extra)


def claude_permission_argv(*, auto_approve: bool) -> List[str]:
    """Full-auto-approve permission argv for non-interactive FCC turns.

    CANONICAL EXPLANATION (Dockerfile ENV IS_SANDBOX=1 comments and the
    orion-harness-governor README point back here -- update this one place,
    not each copy, if Claude Code's behavior changes):

    `--dangerously-skip-permissions` (the raw flag) and `--permission-mode
    bypassPermissions` are the same full-bypass behavior, and BOTH refuse to
    start as root/sudo unless the process recognizes a deliberate sandbox:
    reverse-engineered live 2026-08-13 from the CLI's own bundled source, the
    gate is `getuid()===0 && process.env.IS_SANDBOX!=="1" &&
    !CLAUDE_CODE_BUBBLEWRAP`. Every Dockerfile whose service calls this
    function as root (grep this repo for `claude_permission_argv(` to find
    them) MUST set `ENV IS_SANDBOX=1`, or the claude subprocess crashes on
    startup on every turn -- plain Docker isn't enough, the CLI checks this
    exact env var, not container detection.

    This is NOT the same as `--permission-mode dontAsk`, used here
    previously: dontAsk avoids that startup refusal but is a deny-by-default
    CI mode ("auto-deny every tool call that would otherwise prompt, run
    only permissions.allow-listed / read-only-Bash / PreToolUse-hook-approved
    actions") -- confirmed live 2026-08-13, a headless root FCC turn got
    "Permission to use Bash has been denied because Claude Code is running
    in don't ask mode" on a plain `git commit`, i.e. it silently denied every
    real action instead of auto-approving them.

    Root containers that set IS_SANDBOX=1 for this get FULL Bash/tool access
    with no per-call prompt -- know what else the container mounts (docker
    socket, SSH keys, network mode) before enabling this on a new service.
    """
    if not auto_approve:
        return []
    if os.geteuid() == 0:
        return ["--permission-mode", "bypassPermissions"]
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
