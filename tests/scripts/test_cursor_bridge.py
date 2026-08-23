"""Tests for scripts/hooks/cursor_bridge.py — Cursor ↔ Claude hook dialect."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hooks"))

from cursor_bridge import (  # noqa: E402
    normalize_cursor_to_claude,
    translate_claude_to_cursor,
    run_bridge,
)

BRIDGE = ROOT / "scripts" / "hooks" / "cursor_bridge.py"
HOOKS_JSON = ROOT / ".cursor" / "hooks.json"


def test_normalize_shell_top_level_command() -> None:
    out = normalize_cursor_to_claude({"command": "git reset --hard", "cwd": "/tmp"})
    assert out["tool_input"]["command"] == "git reset --hard"
    assert out["cwd"] == "/tmp"


def test_normalize_conversation_id_to_session_id() -> None:
    out = normalize_cursor_to_claude({"conversation_id": "conv-abc", "status": "completed"})
    assert out["session_id"] == "conv-abc"


def test_normalize_keeps_explicit_session_id() -> None:
    out = normalize_cursor_to_claude(
        {"session_id": "sess-1", "conversation_id": "conv-ignored"}
    )
    assert out["session_id"] == "sess-1"


def test_normalize_write_path_alias() -> None:
    out = normalize_cursor_to_claude(
        {"tool_name": "Write", "tool_input": {"path": "/repo/a.py", "contents": "x"}}
    )
    assert out["tool_input"]["file_path"] == "/repo/a.py"
    assert out["tool_input"]["contents"] == "x"
    assert out["tool_input"]["content"] == "x"


def test_normalize_keeps_existing_file_path() -> None:
    out = normalize_cursor_to_claude(
        {"tool_input": {"file_path": "/keep/me.py", "path": "/ignore/me.py"}}
    )
    assert out["tool_input"]["file_path"] == "/keep/me.py"


def test_translate_deny() -> None:
    claude = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "blocked for reason",
            }
        }
    )
    out = translate_claude_to_cursor(claude, "deny")
    assert out == {
        "permission": "deny",
        "user_message": "blocked for reason",
        "agent_message": "blocked for reason",
    }


def test_translate_deny_empty_allows() -> None:
    assert translate_claude_to_cursor("", "deny") == {"permission": "allow"}


def test_translate_context_and_nudge() -> None:
    claude = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "hello board",
            }
        }
    )
    assert translate_claude_to_cursor(claude, "context") == {
        "additional_context": "hello board"
    }
    assert translate_claude_to_cursor(claude, "nudge") == {
        "additional_context": "hello board"
    }


def test_translate_stop_to_followup() -> None:
    claude = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "Stop",
                "additionalContext": "run checkout",
            }
        }
    )
    assert translate_claude_to_cursor(claude, "stop") == {
        "followup_message": "run checkout"
    }


def test_translate_plain_text_nudge() -> None:
    out = translate_claude_to_cursor("called: hook-guard search", "nudge")
    assert out == {"additional_context": "called: hook-guard search"}


def test_run_bridge_context_joins_parts(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text(
        "import json,sys\n"
        "print(json.dumps({'hookSpecificOutput':{'additionalContext':'one'}}))\n",
        encoding="utf-8",
    )
    b.write_text(
        "import json,sys\n"
        "print(json.dumps({'hookSpecificOutput':{'additionalContext':'two'}}))\n",
        encoding="utf-8",
    )
    result = run_bridge(
        "context",
        [[sys.executable, str(a)], [sys.executable, str(b)]],
        b"{}",
        timeout=5.0,
    )
    assert result == {"additional_context": "one\n\ntwo"}


def test_bridge_cli_skips_stop_followup_on_aborted(tmp_path: Path) -> None:
    stub = tmp_path / "would_nag.py"
    stub.write_text(
        "import json,sys\n"
        "print(json.dumps({"
        "'hookSpecificOutput':{"
        "'hookEventName':'Stop',"
        "'additionalContext':'should-not-emit'"
        "}}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "stop", str(stub)],
        input=json.dumps({"status": "aborted", "conversation_id": "c1"}),
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


def test_bridge_cli_stop_maps_conversation_id(tmp_path: Path) -> None:
    stub = tmp_path / "echo_session.py"
    stub.write_text(
        "import json,sys\n"
        "p=json.load(sys.stdin)\n"
        "print(json.dumps({"
        "'hookSpecificOutput':{"
        "'hookEventName':'Stop',"
        "'additionalContext':p.get('session_id','missing')"
        "}}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "stop", str(stub)],
        input=json.dumps({"status": "completed", "conversation_id": "conv-99"}),
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["followup_message"] == "conv-99"


def test_bridge_cli_deny_with_cursor_shell_shape(tmp_path: Path) -> None:
    """End-to-end: Cursor beforeShellExecution stdin -> deny via real guard.

    Uses a tiny stub that always denies so we don't need a shared checkout.
    """
    stub = tmp_path / "always_deny.py"
    stub.write_text(
        "import json,sys\n"
        "print(json.dumps({"
        "'hookSpecificOutput':{"
        "'hookEventName':'PreToolUse',"
        "'permissionDecision':'deny',"
        "'permissionDecisionReason':'stub-deny'"
        "}}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "deny", str(stub)],
        input=json.dumps({"command": "git reset --hard", "cwd": str(tmp_path)}),
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["permission"] == "deny"
    assert "stub-deny" in payload["agent_message"]


def test_bridge_cli_normalizes_path_for_edit_guard_stub(tmp_path: Path) -> None:
    stub = tmp_path / "echo_path.py"
    stub.write_text(
        "import json,sys\n"
        "p=json.load(sys.stdin)\n"
        "fp=p.get('tool_input',{}).get('file_path','')\n"
        "print(json.dumps({"
        "'hookSpecificOutput':{"
        "'hookEventName':'PreToolUse',"
        "'permissionDecision':'deny',"
        "'permissionDecisionReason':fp"
        "}}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "deny", str(stub)],
        input=json.dumps({"tool_input": {"path": "/mnt/x/file.py"}}),
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["permission"] == "deny"
    assert payload["user_message"] == "/mnt/x/file.py"


def test_cursor_hooks_json_covers_claude_suite() -> None:
    data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    assert data.get("version") == 1
    hooks = data["hooks"]
    rendered = json.dumps(hooks)

    assert "sessionStart" in hooks
    assert "stop" in hooks
    assert "beforeShellExecution" in hooks
    assert "preToolUse" in hooks
    assert "postToolUse" in hooks

    # Same script bodies Claude wires in .claude/settings.json
    for name in (
        "session_start_worktree_summary.py",
        "session_start_agent_board.py",
        "session_start_graph_worktree_guard.py",
        "session_stop_agent_board.py",
        "stop_worktree_wip_snapshot.py",
        "destructive_git_guard.py",
        "shared_checkout_edit_guard.py",
        "graphify_hook_guard_gate.sh",
        "metric_lineage_nudge.py",
        "cursor_bridge.py",
    ):
        assert name in rendered, f"missing {name} in .cursor/hooks.json"

    # Dialect bridge must sit in front of Claude-shaped context/deny/nudge/stop scripts
    assert "cursor_bridge.py context" in rendered
    assert "cursor_bridge.py deny" in rendered
    assert "cursor_bridge.py nudge" in rendered
    assert "cursor_bridge.py stop" in rendered


def test_cursor_hooks_stop_loop_limit_is_one() -> None:
    """Board nag maps to followup_message; keep auto-continue capped at 1."""
    data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    stop_hooks = data["hooks"]["stop"]
    board = [h for h in stop_hooks if "session_stop_agent_board" in h.get("command", "")]
    assert len(board) == 1
    assert board[0].get("loop_limit") == 1


def test_bridge_nudge_through_graphify_gate(tmp_path: Path) -> None:
    fake = tmp_path / "fake_graphify.sh"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\","
        "\"additionalContext\":\"MANDATORY: use graphify\"}}'\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if k != "ORION_FCC_SUBPROCESS"}
    env["GRAPHIFY_HOOK_GUARD_BIN"] = str(fake)

    proc = subprocess.run(
        [
            sys.executable,
            str(BRIDGE),
            "nudge",
            str(ROOT / "scripts/hooks/graphify_hook_guard_gate.sh"),
            "search",
        ],
        input="{}",
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(ROOT),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "graphify" in payload["additional_context"].lower()
