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


def test_thinking_budget_counts_content_not_progress_metadata() -> None:
    assert measure_step_payload_chars({"type": "system", "subtype": "thinking_tokens",
        "estimated_tokens_delta": 1, "uuid": "x" * 1000}) == 0
    assert measure_step_payload_chars({"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "x" * 400},
        {"type": "text", "text": "done"},
    ]}}) == 404


# --- Honest gauge: provider usage, CLI compaction line, degenerate summaries ---

from orion.fcc.context_budget import (  # noqa: E402
    autocompact_trigger_tokens,
    compact_summary_text,
    context_pressure_threshold_chars,
    context_risk_level,
    is_degenerate_compact_summary,
    observed_context_chars,
)

_PREAMBLE = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
)


def _clear_lane_env(monkeypatch) -> None:
    for key in (
        "HARNESS_FCC_MAX_CONTEXT_TOKENS", "HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS",
        "HARNESS_FCC_MAX_OUTPUT_TOKENS", "HUB_AGENT_CLAUDE_MAX_OUTPUT_TOKENS",
        "HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE", "HUB_AGENT_CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
        "HARNESS_FCC_CONTEXT_PRESSURE_PCT", "HUB_AGENT_CLAUDE_CONTEXT_PRESSURE_PCT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ORION_FCC_CHARS_PER_TOKEN", "4")


def test_trigger_model_reproduces_both_live_compaction_clusters(monkeypatch) -> None:
    """154 live auto-compactions at pct=70 and the CLI's 32k default output cap
    clustered at 30-34k (64k lane) and ~76k (128k lane)."""
    _clear_lane_env(monkeypatch)
    monkeypatch.setenv("HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE", "70")
    monkeypatch.setenv("HARNESS_FCC_MAX_OUTPUT_TOKENS", "32000")
    assert 30_000 <= autocompact_trigger_tokens(65536) <= 34_000
    assert 74_000 <= autocompact_trigger_tokens(131072) <= 79_000


def test_shipped_defaults_give_the_64k_lane_more_room(monkeypatch) -> None:
    _clear_lane_env(monkeypatch)
    monkeypatch.setenv("HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE", "90")
    monkeypatch.setenv("HARNESS_FCC_MAX_OUTPUT_TOKENS", "16384")
    assert autocompact_trigger_tokens(65536) == 65536 - 16384 - 13_000
    assert autocompact_trigger_tokens(400) is None


def test_pressure_line_sits_below_the_compaction_line(monkeypatch) -> None:
    """70% of a 64k window (~46k) is past where the CLI compacts (~36k)."""
    _clear_lane_env(monkeypatch)
    monkeypatch.setenv("HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE", "90")
    trigger_chars = autocompact_trigger_tokens(65536) * 4
    assert context_pressure_threshold_chars(65536) < trigger_chars
    assert context_risk_level(accumulated_chars=trigger_chars, max_chars=65536 * 4,
                              compact_at_chars=trigger_chars) == "critical"
    assert context_risk_level(accumulated_chars=trigger_chars, max_chars=65536 * 4) != "critical"


def test_observed_context_reads_provider_usage(monkeypatch) -> None:
    _clear_lane_env(monkeypatch)
    event = {"type": "assistant", "message": {"content": [], "usage": {
        "input_tokens": 30_000, "cache_read_input_tokens": 1_000, "output_tokens": 500}}}
    assert observed_context_chars(event) == 31_500 * 4
    assert observed_context_chars({"type": "assistant", "message": {"usage": {"input_tokens": 0}}}) is None
    assert observed_context_chars({"type": "user", "message": {"usage": {"input_tokens": 9}}}) is None
    assert observed_context_chars({"type": "assistant", "message": {"content": []}}) is None


def test_degenerate_summary_shapes_from_harness_turn_trace() -> None:
    """Openings copied from stored compactions (harness_turn_trace, 2026-10-05..10-17)."""
    lost = [
        "Now let me read the remaining key files to complete the picture.",
        "Good. I found the key files. Now let me read the core source files in parallel.",
        "I cannot provide a traditional summary since this is not a multi-turn development conversation.",
    ]
    kept = "Summary:\n1. Primary Request and Intent:\n   Juniper asked about recent PRs."
    for body in lost:
        text = compact_summary_text({"type": "user", "message": {"content": _PREAMBLE + body}})
        assert text is not None and is_degenerate_compact_summary(text)
    text = compact_summary_text({"type": "user", "message": {"content": [{"type": "text", "text": _PREAMBLE + kept}]}})
    assert text is not None and not is_degenerate_compact_summary(text)
    assert compact_summary_text({"type": "assistant", "message": {"content": "Summary: done"}}) is None
