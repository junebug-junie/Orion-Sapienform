"""Resolve GitHub owner/repo for FCC MCP operator briefs."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple

from orion.fcc.mcp_names import GITHUB, mcp_tool

logger = logging.getLogger("orion.fcc.github_repo_context")

_GIT_SSH = re.compile(r"^git@[^:]+:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")
_GIT_HTTPS = re.compile(r"^https?://[^/]+/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")

_GH_LIST_PRS = mcp_tool(GITHUB, "list_pull_requests")
_GH_GET_PR = mcp_tool(GITHUB, "get_pull_request")
_GH_SEARCH_PRS = mcp_tool(GITHUB, "search_pull_requests")


def harness_mcp_enabled() -> bool:
    return os.environ.get("HARNESS_FCC_MCP_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def default_harness_workspace() -> str:
    return (
        str(os.environ.get("HARNESS_FCC_WORKSPACE") or "").strip()
        or str(os.environ.get("ORION_REPO_ROOT") or "").strip()
        or os.getcwd()
    )


def parse_github_remote_url(url: str) -> Tuple[str, str] | None:
    stripped = str(url or "").strip()
    for pattern in (_GIT_SSH, _GIT_HTTPS):
        match = pattern.match(stripped)
        if match:
            owner = match.group("owner").strip()
            repo = match.group("repo").strip()
            if owner and repo:
                return owner, repo
    return None


def resolve_github_repo_coordinate(*, workspace: Path | str | None = None) -> Tuple[str, str] | None:
    owner = str(os.environ.get("ORION_GITHUB_OWNER") or "").strip()
    repo = str(os.environ.get("ORION_GITHUB_REPO") or "").strip()
    if owner and repo:
        return owner, repo

    combined = str(os.environ.get("ORION_GITHUB_REPOSITORY") or "").strip()
    if "/" in combined:
        parts = [p.strip() for p in combined.split("/", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]

    root = Path(
        str(workspace or "").strip()
        or str(os.environ.get("HARNESS_FCC_WORKSPACE") or "").strip()
        or str(os.environ.get("HUB_AGENT_CLAUDE_WORKSPACE") or "").strip()
        or str(os.environ.get("ORION_REPO_ROOT") or "").strip()
        or os.getcwd()
    )
    try:
        url = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return parse_github_remote_url(url)


def github_mcp_brief_lines(*, workspace: Path | str | None = None) -> list[str]:
    coord = resolve_github_repo_coordinate(workspace=workspace)
    if coord is None:
        return []
    owner, repo = coord
    return [
        (
            f"GitHub MCP is available (owner={owner!r}, repo={repo!r}); its tools are named "
            f"mcp__{GITHUB}__<tool>. "
            "Use it only when this turn's task needs PR/issue/repo facts and you judge "
            "it appropriate — do not fetch GitHub data for unrelated turns."
        ),
        (
            f"If you do query PRs: {_GH_LIST_PRS} requires both owner and repo (never use "
            "the repo name as owner). For the latest PR, pass state=all, sort=updated, "
            "direction=desc, perPage=1 (default state=open returns [] when nothing is open). "
            f"Avoid {_GH_SEARCH_PRS} (blows ~65k ctx). Report the PR title only; never paste "
            "raw MCP JSON into the reply."
        ),
        (
            "When a PR number is already known (user message or imperative), prefer "
            f"{_GH_GET_PR}(owner={owner!r}, repo={repo!r}, pullNumber=N) — never "
            f"{_GH_LIST_PRS} to discover a single named PR."
        ),
        (
            f"Never call {_GH_LIST_PRS} without repo= set. Never omit perPage on list calls; "
            "default perPage=1 unless the task explicitly needs a short ranked list."
        ),
        (
            "If a tool result is truncated by orion-fcc-mcp-proxy, summarize from the excerpt; "
            "do not repeat the same bulk query."
        ),
    ]


def github_mcp_repo_brief_line(*, workspace: Path | str | None = None) -> str | None:
    lines = github_mcp_brief_lines(workspace=workspace)
    return "\n".join(lines) if lines else None


def append_github_mcp_harness_brief(
    parts: list[str],
    *,
    workspace: Path | str | None = None,
) -> None:
    """Append GitHub MCP operator lines when harness MCP is enabled."""
    if not harness_mcp_enabled():
        return
    ws = workspace if workspace is not None else default_harness_workspace()
    lines = github_mcp_brief_lines(workspace=ws)
    if lines:
        parts.extend(lines)
        return
    logger.warning(
        "HARNESS_FCC_MCP_ENABLED but GitHub repo coordinate unresolved "
        "(set ORION_GITHUB_OWNER and ORION_GITHUB_REPO, or ensure git remote "
        "origin in workspace=%s)",
        ws,
    )
