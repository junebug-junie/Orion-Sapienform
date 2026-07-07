from __future__ import annotations

from pathlib import Path

import pytest

from orion.fcc.github_repo_context import (
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


def test_github_mcp_repo_brief_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_GITHUB_OWNER", "junebug-junie")
    monkeypatch.setenv("ORION_GITHUB_REPO", "Orion-Sapienform")
    line = github_mcp_repo_brief_line(workspace=Path("/tmp"))
    assert line is not None
    assert "owner='junebug-junie'" in line
    assert "repo='Orion-Sapienform'" in line
    assert "list_pull_requests" in line
