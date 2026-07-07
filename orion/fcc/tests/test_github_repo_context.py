from __future__ import annotations

from pathlib import Path

import pytest

from orion.fcc.github_repo_context import (
    github_mcp_brief_lines,
    github_mcp_repo_brief_line,
    parse_github_remote_url,
    resolve_github_repo_coordinate,
)


def test_parse_github_remote_url_ssh() -> None:
    assert parse_github_remote_url("git@github.com:junebug-junie/Orion-Sapienform.git") == (
        "junebug-junie",
        "Orion-Sapienform",
    )


def test_parse_github_remote_url_https() -> None:
    assert parse_github_remote_url("https://github.com/acme/widgets.git") == ("acme", "widgets")


def test_resolve_github_repo_coordinate_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_GITHUB_OWNER", "env-owner")
    monkeypatch.setenv("ORION_GITHUB_REPO", "env-repo")
    assert resolve_github_repo_coordinate(workspace="/tmp") == ("env-owner", "env-repo")


def test_github_mcp_brief_lines_include_narrow_pr_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    lines = github_mcp_brief_lines(workspace=Path("/tmp"))
    assert len(lines) == 3
    assert "perPage=1" in lines[1]
    assert "search_pull_requests" in lines[1]
    assert "never paste raw MCP JSON" in lines[2]
