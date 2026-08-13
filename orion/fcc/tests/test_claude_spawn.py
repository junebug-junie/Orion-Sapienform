from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orion.fcc import claude_spawn

REPO_ROOT = Path(__file__).resolve().parents[3]

# Every known production caller of claude_permission_argv(auto_approve=True)
# that runs in a root Docker container (no USER directive), mapped to the
# Dockerfile that MUST set ENV IS_SANDBOX=1 -- see
# claude_permission_argv()'s docstring for why: bypassPermissions refuses to
# start as root without it, so a missing ENV line means the FCC claude
# subprocess crashes on startup on every turn in that container.
#
# scripts/context_mode_hooks_smoke.py is deliberately excluded: it's a
# host-run dev smoke script, not a service container -- no Dockerfile owns it.
_ROOT_CALLERS_TO_DOCKERFILE = {
    "orion/harness/fcc_motor.py": "services/orion-harness-governor/Dockerfile",
    "services/orion-hub/scripts/fcc_claude_bridge.py": "services/orion-hub/Dockerfile",
    "services/orion-self-study-enrichment/app/claude_runner.py": (
        "services/orion-self-study-enrichment/Dockerfile"
    ),
}
_KNOWN_NON_CONTAINER_CALLERS = {"scripts/context_mode_hooks_smoke.py"}


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


def test_claude_permission_argv_root_uses_bypass_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NOT dontAsk: confirmed live 2026-08-13 that dontAsk is a deny-by-default
    # CI mode (denies any tool call, including Bash, that isn't already
    # permissions.allow-listed) -- the opposite of the auto-approve behavior
    # this helper exists to provide for headless root FCC turns.
    monkeypatch.setattr(claude_spawn.os, "geteuid", lambda: 0)
    assert claude_spawn.claude_permission_argv(auto_approve=True) == [
        "--permission-mode",
        "bypassPermissions",
    ]


def test_claude_permission_argv_non_root_uses_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_spawn.os, "geteuid", lambda: 1000)
    assert claude_spawn.claude_permission_argv(auto_approve=True) == [
        "--dangerously-skip-permissions",
    ]


def test_setting_sources_argv_defaults_to_user_local_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_FCC_SETTING_SOURCES", raising=False)
    assert claude_spawn.setting_sources_argv("HARNESS_FCC_SETTING_SOURCES") == [
        "--setting-sources",
        "user,local",
    ]


def test_setting_sources_argv_respects_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_FCC_SETTING_SOURCES", "user,project,local")
    assert claude_spawn.setting_sources_argv("HARNESS_FCC_SETTING_SOURCES") == [
        "--setting-sources",
        "user,project,local",
    ]


def test_setting_sources_argv_explicit_empty_means_no_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present-but-empty is a deliberate opt-out, distinct from unset --
    falls back to Claude Code's own default (all three scopes) instead of
    forcing --setting-sources at all."""
    monkeypatch.setenv("HARNESS_FCC_SETTING_SOURCES", "")
    assert claude_spawn.setting_sources_argv("HARNESS_FCC_SETTING_SOURCES") == []


def test_known_root_callers_dockerfiles_set_is_sandbox() -> None:
    """Every known root-container caller of claude_permission_argv() must
    have ENV IS_SANDBOX=1 in its Dockerfile, or bypassPermissions crashes the
    FCC claude subprocess on startup on every turn (see that function's
    docstring). This locks in today's 3 known pairings so an edit that drops
    the ENV line -- not just a brand-new caller -- is caught too."""
    for caller, dockerfile in _ROOT_CALLERS_TO_DOCKERFILE.items():
        text = (REPO_ROOT / dockerfile).read_text(encoding="utf-8")
        assert re.search(r"^ENV\s+IS_SANDBOX=1\b", text, re.MULTILINE), (
            f"{dockerfile} (owns root caller {caller}) is missing "
            "`ENV IS_SANDBOX=1` -- its FCC claude subprocess will crash on "
            "startup under bypassPermissions"
        )


def test_no_undiscovered_root_callers_of_claude_permission_argv() -> None:
    """Fails loudly if a NEW caller of claude_permission_argv(auto_approve=True)
    shows up anywhere in the repo without being added to
    _ROOT_CALLERS_TO_DOCKERFILE (or _KNOWN_NON_CONTAINER_CALLERS) above --
    the exact recurrence this gate exists to catch: a future service reuses
    this helper as root, tests/review/CI all stay green, and the failure
    only surfaces live as every turn crashing on startup."""
    known = set(_ROOT_CALLERS_TO_DOCKERFILE) | _KNOWN_NON_CONTAINER_CALLERS
    found: set[str] = set()
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if "/tests/" in rel or rel.startswith("tests/") or rel.endswith("/claude_spawn.py"):
            continue
        if ".worktrees/" in rel or "/node_modules/" in rel:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"claude_permission_argv\(\s*auto_approve\s*=\s*True\s*\)", text):
            found.add(rel)
    unknown = found - known
    assert not unknown, (
        f"New claude_permission_argv(auto_approve=True) caller(s) not in "
        f"_ROOT_CALLERS_TO_DOCKERFILE / _KNOWN_NON_CONTAINER_CALLERS: "
        f"{sorted(unknown)} -- if this runs as root in a container, add its "
        "Dockerfile to _ROOT_CALLERS_TO_DOCKERFILE and set ENV IS_SANDBOX=1 "
        "there; if it's host-run (no container), add it to "
        "_KNOWN_NON_CONTAINER_CALLERS instead."
    )
