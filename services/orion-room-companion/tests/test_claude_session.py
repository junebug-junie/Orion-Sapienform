"""Argv construction and result parsing for a room turn."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from app.claude_session import build_argv, run_room_turn

KW = dict(
    claude_bin="claude",
    model="claude-sonnet-5",
    effort="medium",
    setting_sources_env_key="ROOM_COMPANION_SETTING_SOURCES",
)


def test_new_session_uses_session_id_flag():
    argv = build_argv("hi", session_id="abc-123", resume=False, **KW)
    assert "--session-id" in argv
    assert argv[argv.index("--session-id") + 1] == "abc-123"
    assert "--resume" not in argv, "a brand-new session has nothing to resume"


def test_resumed_session_uses_resume_flag():
    argv = build_argv("hi", session_id="abc-123", resume=True, **KW)
    assert "--resume" in argv
    assert argv[argv.index("--resume") + 1] == "abc-123"
    assert "--session-id" not in argv, (
        "passing --session-id alongside --resume is a CLI error, not redundancy"
    )


def test_tools_are_disabled():
    """v1 Claude is a companion to think with, not an agent: no filesystem,
    no repo, no fetch."""
    argv = build_argv("hi", session_id="s", resume=False, **KW)
    assert "--tools" in argv
    assert argv[argv.index("--tools") + 1] == ""


def test_system_prompt_is_appended_when_given():
    argv = build_argv("hi", session_id="s", resume=False, append_system_prompt="be a peer", **KW)
    assert argv[argv.index("--append-system-prompt") + 1] == "be a peer"


def _completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def _run(stdout: str, **kw):
    with patch("app.claude_session.subprocess.run", return_value=_completed(stdout, **kw)):
        return run_room_turn("hi", session_id="s", resume=False, timeout_sec=5, **KW)


def test_successful_turn_extracts_text_cost_model_and_session():
    payload = {
        "is_error": False,
        "result": "Hey Juniper, good to be here.",
        "session_id": "sess-9",
        "total_cost_usd": 0.0084347,
        "modelUsage": {"claude-sonnet-5": {"inputTokens": 10}},
    }
    result = _run(json.dumps(payload))
    assert result.ok
    assert result.text == "Hey Juniper, good to be here."
    assert result.claude_session_id == "sess-9"
    assert result.cost_usd == pytest.approx(0.0084347)
    assert result.model == "claude-sonnet-5"
    assert result.duration_ms >= 0


def test_auth_failure_exits_zero_but_is_not_ok():
    """The failure mode that matters most: an expired token exits 0 and
    returns is_error with a human-readable result. Treating exit code alone
    as truth would publish an empty utterance as a success."""
    payload = {
        "is_error": True,
        "result": "Failed to authenticate. API Error: 401 OAuth access token is invalid.",
        "session_id": "sess-9",
        "total_cost_usd": 0,
    }
    result = _run(json.dumps(payload))
    assert not result.ok
    assert "401" in (result.error or "")
    assert result.text == ""


def test_empty_text_is_a_failure_not_a_quiet_participant():
    """AGENTS.md 0A: raw_len=0 is never success."""
    result = _run(json.dumps({"is_error": False, "result": "   ", "total_cost_usd": 0.01}))
    assert not result.ok
    assert "raw_len=0" in (result.error or "")


def test_unparseable_output_is_a_failure():
    result = _run("not json at all")
    assert not result.ok
    assert "unparseable" in (result.error or "")


def test_nonzero_exit_surfaces_stderr_when_no_json_detail():
    with patch(
        "app.claude_session.subprocess.run",
        return_value=_completed("{}", returncode=2, stderr="claude: boom"),
    ):
        result = run_room_turn("hi", session_id="s", resume=False, timeout_sec=5, **KW)
    assert not result.ok
    assert "boom" in (result.error or "")


def test_timeout_is_reported_not_raised():
    with patch(
        "app.claude_session.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=180),
    ):
        result = run_room_turn("hi", session_id="s", resume=False, timeout_sec=180, **KW)
    assert not result.ok
    assert "timeout" in (result.error or "")


def test_model_falls_back_to_top_level_field():
    result = _run(json.dumps({"result": "hi", "model": "claude-opus-5", "total_cost_usd": 0.02}))
    assert result.model == "claude-opus-5"


def test_model_picks_the_biggest_contributor_not_the_first_key():
    """Regression from the first live smoke run.

    Claude Code bills small background work (title generation and similar) to
    a cheaper model on the same turn, so `modelUsage` legitimately holds more
    than one entry. The first run reported `claude-haiku-4-5` for a turn that
    Sonnet actually answered. Since this field is the guard against a local
    model masquerading as Claude, naming the wrong model is worse than naming
    none -- pick the entry that did the most output work.
    """
    payload = {
        "result": "a real room reply",
        "total_cost_usd": 0.05,
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"outputTokens": 12},
            "claude-sonnet-5": {"outputTokens": 380},
        },
    }
    assert _run(json.dumps(payload)).model == "claude-sonnet-5"


def test_model_handles_usage_entries_without_token_counts():
    payload = {
        "result": "hi",
        "total_cost_usd": 0.01,
        "modelUsage": {"claude-sonnet-5": {}, "claude-haiku-4-5-20251001": None},
    }
    assert _run(json.dumps(payload)).model in {"claude-sonnet-5", "claude-haiku-4-5-20251001"}
