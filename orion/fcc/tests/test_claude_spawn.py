from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.fcc import claude_spawn


def test_mcp_allowed_tool_patterns_per_server() -> None:
    patterns = claude_spawn.mcp_allowed_tool_patterns({"github": {}, "firecrawl": {}})
    assert patterns == ["mcp__github", "mcp__firecrawl"]


def test_mcp_disallowed_blocks_gh_when_github_present() -> None:
    blocked = claude_spawn.mcp_disallowed_tool_patterns({"github": {}, "firecrawl": {}})
    assert blocked == ["Bash(gh *)"]


def test_extend_mcp_argv_uses_per_server_patterns(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"github": {"type": "stdio"}, "firecrawl": {"type": "stdio"}}}),
        encoding="utf-8",
    )
    argv: list[str] = ["claude", "-p", "hi"]
    claude_spawn.extend_mcp_argv(argv, cfg)
    assert argv == [
        "claude",
        "-p",
        "hi",
        "--mcp-config",
        str(cfg),
        "--allowedTools",
        "mcp__github",
        "mcp__firecrawl",
        "--disallowedTools",
        "Bash(gh *)",
    ]


def test_extend_mcp_argv_appends_extra_allowed_tools_after_server_patterns(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"github": {"type": "stdio"}, "firecrawl": {"type": "stdio"}}}),
        encoding="utf-8",
    )
    argv: list[str] = ["claude", "-p", "hi"]
    claude_spawn.extend_mcp_argv(
        argv, cfg, extra_allowed_tools=["mcp__plugin_context-mode_context-mode"]
    )
    assert argv == [
        "claude",
        "-p",
        "hi",
        "--mcp-config",
        str(cfg),
        "--allowedTools",
        "mcp__github",
        "mcp__firecrawl",
        "mcp__plugin_context-mode_context-mode",
        "--disallowedTools",
        "Bash(gh *)",
    ]


def test_extend_mcp_argv_extra_none_or_empty_keeps_argv_identical(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"github": {"type": "stdio"}, "firecrawl": {"type": "stdio"}}}),
        encoding="utf-8",
    )
    baseline: list[str] = ["claude", "-p", "hi"]
    claude_spawn.extend_mcp_argv(baseline, cfg)
    with_none: list[str] = ["claude", "-p", "hi"]
    claude_spawn.extend_mcp_argv(with_none, cfg, extra_allowed_tools=None)
    with_empty: list[str] = ["claude", "-p", "hi"]
    claude_spawn.extend_mcp_argv(with_empty, cfg, extra_allowed_tools=[])
    assert with_none == baseline
    assert with_empty == baseline


def test_extend_mcp_argv_extra_with_zero_servers_still_emits_allowed_tools(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    argv: list[str] = ["claude", "-p", "hi"]
    claude_spawn.extend_mcp_argv(
        argv, cfg, extra_allowed_tools=["mcp__plugin_context-mode_context-mode"]
    )
    assert argv == [
        "claude",
        "-p",
        "hi",
        "--mcp-config",
        str(cfg),
        "--allowedTools",
        "mcp__plugin_context-mode_context-mode",
    ]


def test_extend_mcp_argv_zero_servers_no_extra_omits_allowed_tools(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    argv: list[str] = ["claude", "-p", "hi"]
    claude_spawn.extend_mcp_argv(argv, cfg)
    assert argv == ["claude", "-p", "hi", "--mcp-config", str(cfg)]


def test_claude_permission_argv_root_uses_dont_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_spawn.os, "geteuid", lambda: 0)
    assert claude_spawn.claude_permission_argv(auto_approve=True) == [
        "--permission-mode",
        "dontAsk",
    ]


def test_claude_permission_argv_non_root_uses_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_spawn.os, "geteuid", lambda: 1000)
    assert claude_spawn.claude_permission_argv(auto_approve=True) == [
        "--dangerously-skip-permissions",
    ]
