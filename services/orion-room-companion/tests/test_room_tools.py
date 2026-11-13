"""Orion's tools in a room turn (Patch 1b): the exact argv, the per-turn MCP config, the footer under
Claude's reply, and that consults and the off switch never get them.

Acceptance check 4 of docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md: a
room turn's argv has --tools "", --strict-mcp-config and only the orion_room server allowed; a consult
turn's argv is unchanged.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from app.claude_session import ClaudeTurnResult, build_argv
from app.main import build_subprocess_env, run_turn
from app.room_prompt import TOOLS_CLAUSE, append_system_prompt_for
from app.room_tools import footer_for, mcp_config, prepare_turn_tools, sweep_stale
from app.settings import Settings
from orion.schemas.claude_tools import ClaudeToolBindingV1
from orion.schemas.room_claude import RoomClaudeRequestV1

KW = dict(claude_bin="claude", model="claude-sonnet-5", effort="medium",
          setting_sources_env_key="ROOM_COMPANION_SETTING_SOURCES")


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        ROOM_COMPANION_SESSION_STATE_PATH=str(tmp_path / "sessions.json"),
        ROOM_COMPANION_CLAUDE_CONFIG_DIR="/root/.claude",
        ROOM_COMPANION_WORKSPACE=str(tmp_path / "workspace"),
        ROOM_COMPANION_TOOLS_STATE_DIR=str(tmp_path / "tools"),
        ORION_BUS_URL="redis://100.92.216.81:6379/0",
    )


def _request(**kw) -> RoomClaudeRequestV1:
    base = dict(room_id="hub-direct", invited_by="Juniper", prompt="@claude is the meter blocking consults?",
                session_id="hub-sess-1")
    base.update(kw)
    return RoomClaudeRequestV1(**base)


def _ok(text="Checked: it is not.") -> ClaudeTurnResult:
    return ClaudeTurnResult(ok=True, text=text, claude_session_id="sess-1", model="claude-sonnet-5",
                            cost_usd=0.01, duration_ms=900, exit_code=0)


def test_room_turn_argv_is_no_builtin_tools_plus_one_strict_mcp_server(tmp_path):
    cfg = tmp_path / "t.mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"orion_room": {"command": "python"}}}))
    argv = build_argv(session_id="s", resume=False, mcp_config_path=cfg, **KW)
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--mcp-config") + 1] == str(cfg)
    assert "--strict-mcp-config" in argv
    i = argv.index("--allowedTools")
    allowed = []
    for tok in argv[i + 1:]:
        if tok.startswith("--"):
            break
        allowed.append(tok)
    assert allowed == ["mcp__orion_room"], "only Orion's room server; no built-in tool is allowed"
    assert "--disallowedTools" not in argv and "Bash" not in " ".join(argv)


def test_consult_argv_is_unchanged_even_if_a_config_is_passed(tmp_path):
    cfg = tmp_path / "t.mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"orion_room": {}}}))
    plain = build_argv(session_id="s", resume=False, tools="Read,Grep,Glob", consult=True, **KW)
    with_cfg = build_argv(session_id="s", resume=False, tools="Read,Grep,Glob", consult=True, mcp_config_path=cfg, **KW)
    assert plain == with_cfg and "--mcp-config" not in with_cfg


def test_no_config_means_the_old_argv():
    argv = build_argv(session_id="s", resume=False, **KW)
    assert "--mcp-config" not in argv and "--strict-mcp-config" not in argv and "--allowedTools" not in argv


def test_turn_config_is_private_and_carries_a_server_authored_binding(settings):
    tools = prepare_turn_tools(settings, _request(), claude_session_id="sess-1")
    mode = stat.S_IMODE(os.stat(tools.mcp_config_path).st_mode)
    assert mode == 0o600
    server = json.loads(tools.mcp_config_path.read_text())["mcpServers"]["orion_room"]
    assert server["args"] == ["-P", "-m", "orion.claude_tools.mcp_server"]
    assert server["env"]["ORION_BUS_URL"] == "redis://100.92.216.81:6379/0"
    binding = ClaudeToolBindingV1.model_validate_json(server["env"]["ORION_CLAUDE_TOOLS_BINDING"])
    assert (binding.room_id, binding.claude_session_id, binding.hub_session_id, binding.topic_from) == (
        "hub-direct", "sess-1", "hub-sess-1", "Juniper")
    assert binding.topic.startswith("@claude is the meter")
    assert binding.call_log_path == str(tools.call_log_path)
    # Nothing in the config a tool could use as a credential.
    assert "TOKEN" not in tools.mcp_config_path.read_text().upper()


def test_run_turn_wires_tools_appends_the_footer_and_cleans_up(settings):
    seen = {}

    def fake_turn(prompt, **kw):
        seen.update(kw)
        log = json.loads(Path(kw["mcp_config_path"]).read_text())["mcpServers"]["orion_room"]["env"]
        path = ClaudeToolBindingV1.model_validate_json(log["ORION_CLAUDE_TOOLS_BINDING"]).call_log_path
        Path(path).write_text(json.dumps({"tool": "body_economy", "decision_path": "asked", "orion_answer": "yes",
                                          "orion_reason": "my costs are fine to share"}) + "\n")
        return _ok()

    with patch("app.main.run_room_turn", side_effect=fake_turn):
        utt = run_turn(settings, _request())
    assert utt.ok and utt.text == ('Checked: it is not.\n\nLooked at: body economy (Orion: yes, '
                                   '"my costs are fine to share").')
    assert TOOLS_CLAUSE in seen["append_system_prompt"]
    assert seen["env"]["MCP_TOOL_TIMEOUT"] == "70000"
    assert not Path(seen["mcp_config_path"]).exists(), "per-turn files are deleted after the turn"
    assert list(Path(settings.ROOM_COMPANION_TOOLS_STATE_DIR).iterdir()) == []


def test_no_calls_no_footer(settings):
    with patch("app.main.run_room_turn", return_value=_ok()):
        assert run_turn(settings, _request()).text == "Checked: it is not."


def test_a_pass_stays_silent_even_after_lookups(settings):
    def fake_turn(prompt, **kw):
        env = json.loads(Path(kw["mcp_config_path"]).read_text())["mcpServers"]["orion_room"]["env"]
        path = ClaudeToolBindingV1.model_validate_json(env["ORION_CLAUDE_TOOLS_BINDING"]).call_log_path
        Path(path).write_text(json.dumps({"tool": "prime_now", "decision_path": "standing_allow"}) + "\n")
        return _ok("[pass]")

    with patch("app.main.run_room_turn", side_effect=fake_turn):
        utt = run_turn(settings, _request(trigger="auto"))
    assert utt.passed and utt.text == ""


def test_consult_rooms_never_get_tools(settings):
    seen = {}
    with patch("app.main.run_room_turn", side_effect=lambda prompt, **kw: seen.update(kw) or _ok()):
        run_turn(settings, _request(room_id="consult:abc123"))
    assert seen["mcp_config_path"] is None and "MCP_TOOL_TIMEOUT" not in seen["env"]
    assert TOOLS_CLAUSE not in seen["append_system_prompt"]


def test_off_switch_is_the_old_turn(settings):
    off = settings.model_copy(update={"ROOM_COMPANION_TOOLS_ENABLED": False})
    seen = {}
    with patch("app.main.run_room_turn", side_effect=lambda prompt, **kw: seen.update(kw) or _ok()):
        run_turn(off, _request())
    assert seen["mcp_config_path"] is None and TOOLS_CLAUSE not in seen["append_system_prompt"]


def test_a_tools_setup_failure_runs_the_turn_without_tools(settings):
    bad = settings.model_copy(update={"ROOM_COMPANION_TOOLS_STATE_DIR": "/proc/definitely/not/writable"})
    seen = {}
    with patch("app.main.run_room_turn", side_effect=lambda prompt, **kw: seen.update(kw) or _ok()):
        assert run_turn(bad, _request()).ok
    assert seen["mcp_config_path"] is None


def test_subprocess_env_never_gets_the_bus_url_or_a_credential_from_tools(settings, monkeypatch):
    monkeypatch.setenv("ORION_BUS_URL", "redis://x:6379/0")
    env = build_subprocess_env(settings, tools=True)
    assert "ORION_BUS_URL" not in env and env["MCP_TOOL_TIMEOUT"] == "70000"


def test_stale_turn_files_are_swept(tmp_path):
    old = tmp_path / "x.mcp.json"
    old.write_text("{}")
    os.utime(old, (0, 0))
    fresh = tmp_path / "y.calls.jsonl"
    fresh.write_text("")
    assert sweep_stale(tmp_path) == 1 and not old.exists() and fresh.exists()


def test_footer_reads_only_well_formed_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('not json\n{"tool": "prime_now", "decision_path": "standing_deny"}\n')
    assert footer_for(p) == "Looked at: prime now (Orion: no, standing)."
    assert footer_for(tmp_path / "missing.jsonl") is None


def test_tools_clause_only_for_tool_turns():
    assert TOOLS_CLAUSE in append_system_prompt_for(trigger="manual", prompt="hi", tools=True)
    assert TOOLS_CLAUSE not in append_system_prompt_for(trigger="manual", prompt="hi")


def test_mcp_config_shape():
    b = ClaudeToolBindingV1(turn_request_id="r", room_id="room", platform="hub", claude_session_id="s")
    cfg = mcp_config(b, python="/usr/local/bin/python", pythonpath="/app", bus_url="redis://h:6379/0")
    assert list(cfg["mcpServers"]) == ["orion_room"]
    assert cfg["mcpServers"]["orion_room"]["command"] == "/usr/local/bin/python"
