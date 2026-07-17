from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "agent_board.py"


def _run(cwd: Path, board_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)}
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_add_and_list_this_worktree_item(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    add = _run(
        primary_repo,
        board,
        "add",
        "--kind",
        "finding",
        "--severity",
        "should",
        "--summary",
        "Review found an unaddressed edge case.",
        "--files",
        "scripts/agent_board.py",
    )
    assert add.returncode == 0, add.stderr
    assert "item:" in add.stdout

    listed = _run(primary_repo, board, "list", "--worktree", str(primary_repo))

    assert listed.returncode == 0
    assert "Review found an unaddressed edge case." in listed.stdout
    assert "scripts/agent_board.py" in listed.stdout


def test_add_rejects_juniper_scope_without_note(primary_repo: Path, tmp_path: Path) -> None:
    proc = _run(
        primary_repo,
        tmp_path / "agent-board.jsonl",
        "add",
        "--kind",
        "decision",
        "--severity",
        "blocker",
        "--scope",
        "juniper",
        "--summary",
        "Needs owner decision.",
    )

    assert proc.returncode == 2
    assert "scope_note is required" in proc.stderr


def test_heartbeat_updates_presence(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    heartbeat = _run(
        primary_repo,
        board,
        "heartbeat",
        "--summary",
        "Building the board CLI.",
        "--task",
        "Task 2 CLI commands.",
    )
    assert heartbeat.returncode == 0

    listed = _run(primary_repo, board, "list", "--all")

    assert "Building the board CLI." in listed.stdout
    assert "Task 2 CLI commands." in listed.stdout


def test_resolve_marks_item_resolved(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    add = _run(
        primary_repo,
        board,
        "add",
        "--kind",
        "followup",
        "--severity",
        "note",
        "--summary",
        "Close after verification.",
    )
    item_id = add.stdout.strip().split("item:", 1)[1].strip()

    resolved = _run(primary_repo, board, "resolve", item_id)
    listed = _run(primary_repo, board, "list", "--all")

    assert resolved.returncode == 0
    assert "resolved" in listed.stdout


def test_checkout_closes_presence_but_does_not_fail_on_open_items(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    _run(primary_repo, board, "heartbeat", "--summary", "Short session.", "--task", "Parking an item.")
    _run(
        primary_repo,
        board,
        "add",
        "--kind",
        "finding",
        "--severity",
        "note",
        "--summary",
        "Open but non-blocking.",
    )

    checkout = _run(primary_repo, board, "checkout")
    listed = _run(primary_repo, board, "list", "--all")

    assert checkout.returncode == 0
    assert "Open items remain" in checkout.stdout
    assert "closed" in listed.stdout
