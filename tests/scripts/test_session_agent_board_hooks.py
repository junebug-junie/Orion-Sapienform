from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_HOOK = ROOT / "scripts" / "hooks" / "session_start_agent_board.py"
STOP_HOOK = ROOT / "scripts" / "hooks" / "session_stop_agent_board.py"


def _run_hook(hook: Path, cwd: Path, board_path: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)}
    return subprocess.run(
        [sys.executable, str(hook)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_session_start_hook_emits_valid_json(primary_repo: Path, tmp_path: Path) -> None:
    proc = _run_hook(START_HOOK, primary_repo, tmp_path / "agent-board.jsonl")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Agent workspace board" in payload["hookSpecificOutput"]["additionalContext"]


def test_session_stop_hook_emits_checkout_context(primary_repo: Path, tmp_path: Path) -> None:
    proc = _run_hook(STOP_HOOK, primary_repo, tmp_path / "agent-board.jsonl")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "Stop"
    assert "agent board checkout" in payload["hookSpecificOutput"]["additionalContext"].lower()


def test_hooks_fail_open_outside_git_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()

    start = _run_hook(START_HOOK, outside, tmp_path / "agent-board.jsonl")
    stop = _run_hook(STOP_HOOK, outside, tmp_path / "agent-board.jsonl")

    assert start.returncode == 0
    assert start.stdout == ""
    assert stop.returncode == 0
    assert stop.stdout == ""


def test_cursor_hooks_file_calls_agent_board_scripts() -> None:
    hooks_path = ROOT / ".cursor" / "hooks.json"
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    rendered = json.dumps(data)
    assert "sessionStart" in rendered
    assert "stop" in rendered
    assert "session_start_agent_board.py" in rendered
    assert "session_stop_agent_board.py" in rendered
