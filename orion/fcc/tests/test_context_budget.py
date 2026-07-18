from __future__ import annotations

import json

import pytest

from orion.fcc.context_budget import (
    CONTEXT_PRESSURE_NUDGE,
    annotate_harness_step,
    apply_context_overflow_hint,
    build_context_pressure_step,
    context_risk_level,
    extend_fcc_subprocess_env,
    is_context_overflow_text,
    measure_step_payload_chars,
    orion_fcc_claude_config_dir,
)
from orion.fcc.mcp_stdio_proxy import _maybe_truncate_line


def test_is_context_overflow_detects_prompt_too_long() -> None:
    assert is_context_overflow_text("Prompt is too long") is True
    assert is_context_overflow_text("exceed_context_size_error") is True


def test_apply_context_overflow_hint_appends_operator_guidance() -> None:
    out = apply_context_overflow_hint("Prompt is too long", n_ctx=65536)
    assert "get_pull_request" in out
    assert "perPage=1" in out


def test_measure_tool_result_step_chars() -> None:
    step = {
        "type": "user",
        "raw": {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "x" * 20_000},
                ]
            },
        },
    }
    assert measure_step_payload_chars(step) == 20_000
    assert context_risk_level(accumulated_chars=40_000, step_chars=20_000) == "critical"


def test_annotate_harness_step_adds_context_obs() -> None:
    step = {"type": "assistant", "raw": {"type": "assistant", "message": {"content": []}}}
    annotated = annotate_harness_step(step, accumulated_chars=50_000)
    assert "context_obs" in annotated
    assert annotated["context_obs"]["risk"] in {"ok", "warn", "critical"}


def test_build_context_pressure_step_carries_nudge() -> None:
    step = build_context_pressure_step(fill_pct=72)
    assert step["raw"]["message"] == CONTEXT_PRESSURE_NUDGE
    assert step["raw"]["context_fill_pct"] == 72


def test_mcp_proxy_truncates_large_tool_text() -> None:
    huge = "a" * 20_000
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": huge}],
            },
        }
    ) + "\n"
    out = _maybe_truncate_line(line, max_chars=12_000)
    parsed = json.loads(out.strip())
    text = parsed["result"]["content"][0]["text"]
    assert len(text) < len(huge)
    assert "orion-fcc-mcp-proxy" in text


def test_extend_fcc_subprocess_env_sets_enable_tool_search_when_unset() -> None:
    env: dict[str, str] = {}
    extend_fcc_subprocess_env(env)
    assert env["ENABLE_TOOL_SEARCH"] == "true"


def test_extend_fcc_subprocess_env_preserves_explicit_tool_search_override() -> None:
    env = {"ENABLE_TOOL_SEARCH": "false"}
    extend_fcc_subprocess_env(env)
    assert env["ENABLE_TOOL_SEARCH"] == "false"


def test_extend_fcc_subprocess_env_preserves_auto_tool_search_override() -> None:
    env = {"ENABLE_TOOL_SEARCH": "auto:5"}
    extend_fcc_subprocess_env(env)
    assert env["ENABLE_TOOL_SEARCH"] == "auto:5"


def test_orion_fcc_claude_config_dir_returns_none_when_unconfigured(monkeypatch) -> None:
    """Neither key set -> inherit whatever the deployment already provides
    (e.g. orion-harness-governor's harness-claude-config volume at
    /root/.claude) rather than forcing an override."""
    monkeypatch.delenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HUB_AGENT_CLAUDE_CONFIG_DIR", raising=False)
    assert orion_fcc_claude_config_dir() is None


def test_orion_fcc_claude_config_dir_harness_set_but_empty_means_inherit(monkeypatch) -> None:
    """orion-harness-governor's .env_example ships this key present but
    empty by default -- an explicit "don't override" signal, distinct from
    the key being absent entirely."""
    monkeypatch.setenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", "")
    monkeypatch.setenv("HUB_AGENT_CLAUDE_CONFIG_DIR", "/tmp/hub-claude-config")
    assert orion_fcc_claude_config_dir() is None


def test_orion_fcc_claude_config_dir_prefers_harness_env(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", "/tmp/harness-claude-config")
    monkeypatch.setenv("HUB_AGENT_CLAUDE_CONFIG_DIR", "/tmp/hub-claude-config")
    assert orion_fcc_claude_config_dir() == "/tmp/harness-claude-config"


def test_orion_fcc_claude_config_dir_falls_back_to_hub_env(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HUB_AGENT_CLAUDE_CONFIG_DIR", "/tmp/hub-claude-config")
    assert orion_fcc_claude_config_dir() == "/tmp/hub-claude-config"


def test_orion_fcc_claude_config_dir_hub_default_is_not_nested_under_dot_fcc(monkeypatch) -> None:
    """Regression guard for orion-hub's real .env_example default value:
    both orion-hub and orion-harness-governor bind-mount
    ${HOME}/.fcc:/root/.fcc:ro (operator secrets). A value nested under
    ~/.fcc would be unwritable by `claude` from inside either container --
    confirmed live (mkdir/touch both fail with "Read-only file system")."""
    monkeypatch.delenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HUB_AGENT_CLAUDE_CONFIG_DIR", "~/.claude-fcc")
    resolved = orion_fcc_claude_config_dir()
    assert resolved is not None
    assert "/.fcc/" not in resolved


def test_extend_fcc_subprocess_env_leaves_claude_config_dir_untouched_when_unconfigured(
    monkeypatch,
) -> None:
    """orion-harness-governor's real deployment: neither key configured ->
    don't set CLAUDE_CONFIG_DIR at all, so claude falls through to its own
    default (the harness-claude-config volume at /root/.claude)."""
    monkeypatch.delenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HUB_AGENT_CLAUDE_CONFIG_DIR", raising=False)
    env: dict[str, str] = {}
    extend_fcc_subprocess_env(env)
    assert "CLAUDE_CONFIG_DIR" not in env


def test_extend_fcc_subprocess_env_sets_and_creates_claude_config_dir(monkeypatch, tmp_path) -> None:
    config_dir = tmp_path / "claude-config"
    monkeypatch.setenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", str(config_dir))
    assert not config_dir.exists()
    env: dict[str, str] = {}
    extend_fcc_subprocess_env(env)
    assert env["CLAUDE_CONFIG_DIR"] == str(config_dir)
    assert config_dir.is_dir()


def test_extend_fcc_subprocess_env_preserves_explicit_claude_config_dir_override(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", str(tmp_path / "unused"))
    operator_dir = str(tmp_path / "operator-chosen")
    env = {"CLAUDE_CONFIG_DIR": operator_dir}
    extend_fcc_subprocess_env(env)
    assert env["CLAUDE_CONFIG_DIR"] == operator_dir
