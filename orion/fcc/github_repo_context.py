"""Resolve GitHub owner/repo for FCC MCP operator briefs."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Tuple

_GIT_SSH = re.compile(r"^git@[^:]+:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")
_GIT_HTTPS = re.compile(r"^https?://[^/]+/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")


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


def github_mcp_repo_brief_line(*, workspace: Path | str | None = None) -> str | None:
    coord = resolve_github_repo_coordinate(workspace=workspace)
    if coord is None:
        return None
    owner, repo = coord
    return (
        f"GitHub MCP: pass owner={owner!r} and repo={repo!r} to list_pull_requests "
        f"(both required; never use the repo name as owner)."
    )
