"""Consult rooms: Claude reads code read-only, thinks harder, never passes, and never bypasses.

The permission premise (default mode + deny rules blocks every read outside the working
directory, Grep and Glob included) was verified live in this container on 2026-11-11 with
claude 2.1.239; these tests pin the argv that premise depends on.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.claude_session import CONSULT_DENY_RULES, ClaudeTurnResult, build_argv
from app.main import run_turn
from app.room_prompt import SYSTEM_PROMPT
from app.settings import Settings
from orion.consult.core import CLAUDE_CONSULT_SYSTEM, consult_room_id
from orion.schemas.room_claude import RoomClaudeRequestV1


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        ROOM_COMPANION_SESSION_STATE_PATH=str(tmp_path / "sessions.json"),
        ROOM_COMPANION_CLAUDE_CONFIG_DIR="/root/.claude",
        ROOM_COMPANION_WORKSPACE=str(tmp_path / "workspace"),
        ROOM_COMPANION_CONSULT_REPO="/consult-src/current",
    )


def _ok(text="hi") -> ClaudeTurnResult:
    return ClaudeTurnResult(ok=True, text=text, claude_session_id="s1", model="claude-sonnet-5-5",
                            cost_usd=0.2, duration_ms=10, exit_code=0)


def _argv(consult: bool) -> list[str]:
    return build_argv(claude_bin="claude", model="m", effort="high", setting_sources_env_key="X",
                      session_id="s", resume=False, tools="Read,Grep,Glob", consult=consult)


def test_consult_argv_has_read_tools_and_never_bypasses():
    argv = _argv(True)
    assert argv[argv.index("--tools") + 1] == "Read,Grep,Glob"
    assert argv[argv.index("--permission-mode") + 1] == "default"
    assert "bypassPermissions" not in argv and "--dangerously-skip-permissions" not in argv
    deny = json.loads(argv[argv.index("--settings") + 1])["permissions"]["deny"]
    assert deny == list(CONSULT_DENY_RULES)
    assert "Read(//proc/**)" in deny and "Read(//root/**)" in deny


def test_companion_argv_still_has_no_tools():
    argv = _argv(False)
    assert argv[argv.index("--tools") + 1] == ""
    assert "--settings" not in argv


def test_consult_room_turn_uses_consult_model_repo_cwd_and_framing(settings):
    calls = []

    def _capture(prompt, **kw):
        calls.append((prompt, kw))
        return _ok()

    req = RoomClaudeRequestV1(room_id=consult_room_id("c1"), invited_by="Orion", prompt="Orion:\nwhy?",
                              trigger="auto")
    with patch("app.main.run_room_turn", side_effect=_capture):
        utt = run_turn(settings, req)
    prompt, kw = calls[0]
    assert prompt == "Orion:\nwhy?", "Hub writes the consult prompt; the companion must not decorate it"
    assert kw["consult"] is True and kw["tools"] == "Read,Grep,Glob"
    assert kw["cwd"] == "/consult-src/current"
    assert kw["model"] == settings.ROOM_COMPANION_CONSULT_MODEL
    assert kw["effort"] == "high"
    assert kw["append_system_prompt"] == SYSTEM_PROMPT + CLAUDE_CONSULT_SYSTEM
    assert utt.ok and utt.text == "hi"


def test_consult_reply_is_never_a_pass(settings):
    req = RoomClaudeRequestV1(room_id=consult_room_id("c2"), invited_by="Orion", prompt="x", trigger="auto")
    with patch("app.main.run_room_turn", return_value=_ok("[pass]")):
        utt = run_turn(settings, req)
    assert utt.passed is False and utt.text == "[pass]"


def test_social_room_turn_is_unchanged(settings):
    calls = []
    with patch("app.main.run_room_turn", side_effect=lambda p, **kw: calls.append(kw) or _ok()):
        run_turn(settings, RoomClaudeRequestV1(room_id="hub-direct", invited_by="Juniper", prompt="hey"))
    assert calls[0]["consult"] is False and calls[0]["tools"] == ""
    assert calls[0]["cwd"] == settings.ROOM_COMPANION_WORKSPACE
