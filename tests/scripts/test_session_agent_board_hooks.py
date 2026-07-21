from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_HOOK = ROOT / "scripts" / "hooks" / "session_start_agent_board.py"
STOP_HOOK = ROOT / "scripts" / "hooks" / "session_stop_agent_board.py"


def _run_hook(
    hook: Path, cwd: Path, board_path: Path, *, stdin_payload: dict | None = None
) -> subprocess.CompletedProcess:
    # Strip the real session id this test process may itself be running
    # under (e.g. a live Claude Code session) so tests only see the
    # session_id explicitly passed via stdin_payload, not ambient env.
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)}
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    # Explicit input= regardless of whether a real payload is given -- these
    # hooks now read stdin (for session_id), and relying on however pytest
    # happens to have stdin configured is fragile; an empty string mimics
    # "no payload" the same way a closed stdin would.
    stdin_text = json.dumps(stdin_payload) if stdin_payload is not None else ""
    return subprocess.run(
        [sys.executable, str(hook)],
        cwd=cwd,
        env=env,
        input=stdin_text,
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
    board = tmp_path / "agent-board.jsonl"
    add = _run_board_cli(
        primary_repo, board, "add", "--kind", "finding", "--severity", "note", "--summary", "found something"
    )
    assert add.returncode == 0, add.stderr

    proc = _run_hook(STOP_HOOK, primary_repo, board)

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "Stop"
    assert "agent board checkout" in payload["hookSpecificOutput"]["additionalContext"].lower()


def test_session_stop_hook_silent_when_no_open_items(primary_repo: Path, tmp_path: Path) -> None:
    """No open items means nothing actionable -- announcing that on every
    single Stop is pure token noise, not a reminder. Stay silent instead of
    emitting a "no open board items" message every turn."""
    proc = _run_hook(STOP_HOOK, primary_repo, tmp_path / "agent-board.jsonl")

    assert proc.returncode == 0
    assert proc.stdout == ""


def test_hooks_noop_for_fcc_subprocess(primary_repo: Path, tmp_path: Path) -> None:
    """orion/harness/fcc_motor.py spawns `claude -p` with cwd set to this
    repo checkout, so it picks up .claude/settings.json's SessionStart/Stop
    hooks the same as a real interactive session -- without this gate, every
    Orion chat turn would write a spurious presence row to the shared board
    and add noise to the FCC harness step count. fcc_motor.py tags its
    subprocess env with ORION_FCC_SUBPROCESS=1 for exactly this check."""
    board = tmp_path / "agent-board.jsonl"
    # Strip the real session id this test process may itself be running
    # under -- harmless today since the ORION_FCC_SUBPROCESS gate returns
    # before either hook reads session_id, but kept consistent with the
    # other helpers in this file so it stays harmless if that changes.
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board), "ORION_FCC_SUBPROCESS": "1"}
    env.pop("CLAUDE_CODE_SESSION_ID", None)

    start = subprocess.run(
        [sys.executable, str(START_HOOK)],
        cwd=primary_repo,
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=30,
    )
    stop = subprocess.run(
        [sys.executable, str(STOP_HOOK)],
        cwd=primary_repo,
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert start.returncode == 0
    assert start.stdout == ""
    assert stop.returncode == 0
    assert stop.stdout == ""
    assert not board.exists(), "FCC subprocess must not write a board presence row"


def test_hooks_fail_open_outside_git_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()

    start = _run_hook(START_HOOK, outside, tmp_path / "agent-board.jsonl")
    stop = _run_hook(STOP_HOOK, outside, tmp_path / "agent-board.jsonl")

    assert start.returncode == 0
    assert start.stdout == ""
    assert stop.returncode == 0
    assert stop.stdout == ""


def _run_board_cli(cwd: Path, board_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)}
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agent_board.py"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_stop_hook_uses_session_id_to_find_the_real_worktree(
    repo_with_worktrees: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Regression test for a real bug caught live: Claude Code's Stop hook
    runs with a process cwd fixed to wherever the session originally
    started, which does NOT track `cd` calls a Bash tool made mid-session --
    confirmed live by inspecting the hook's own stdin payload, which reported
    `cwd` as the shared/primary checkout even while real git work had been
    happening in a linked worktree for many turns. `git rev-parse
    --show-toplevel` from that same fixed cwd resolves to the same wrong
    answer, so the hook was reporting (or missing) open items for the wrong
    worktree entirely.

    A git-hook-driven heartbeat (`scripts/git_hooks/post-commit`) tags its
    heartbeat with `$CLAUDE_CODE_SESSION_ID` when set -- that hook DOES run
    with the correct cwd (invoked by `git`, not the Claude Code harness). The
    Stop hook can then look up "the most recently heartbeated worktree for
    this session_id" from its own stdin payload instead of trusting its own
    cwd."""
    primary, merged_wt, _unmerged_wt = repo_with_worktrees
    board = tmp_path / "agent-board.jsonl"
    session_id = "sess-abc-123"

    # Simulate the git-hook-driven heartbeat that happens on a real commit
    # in merged_wt, tagged with the session id.
    heartbeat = _run_board_cli(
        merged_wt, board, "heartbeat", "--summary", "real work happened here", "--session-id", session_id
    )
    assert heartbeat.returncode == 0, heartbeat.stderr

    # An open item scoped to merged_wt -- this is what the Stop hook should
    # report on, since that's where the session's real work is.
    add = _run_board_cli(
        merged_wt, board, "add", "--kind", "finding", "--severity", "note", "--summary", "found something"
    )
    assert add.returncode == 0, add.stderr

    # Run the Stop hook from `primary` (simulating the harness's fixed,
    # wrong cwd) but WITH the session_id in its stdin payload.
    proc = _run_hook(STOP_HOOK, primary, board, stdin_payload={"session_id": session_id})
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    detail = payload["hookSpecificOutput"]["additionalContext"]
    assert "1 open item" in detail, detail


def test_stop_hook_falls_back_to_cwd_when_session_id_has_no_match(
    primary_repo: Path, tmp_path: Path
) -> None:
    """No git-hook-driven heartbeat has landed for this session_id yet (e.g.
    the very first hook fire in a session) -- must fall back to the old
    git-rev-parse-based resolution rather than erroring or misreporting.
    No open items exist for the fallback-resolved worktree, so the hook
    stays silent (see test_session_stop_hook_silent_when_no_open_items)."""
    board = tmp_path / "agent-board.jsonl"
    proc = _run_hook(STOP_HOOK, primary_repo, board, stdin_payload={"session_id": "no-such-session"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_session_start_hook_resolves_via_session_id_not_fixed_cwd(
    repo_with_worktrees: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Same bug, SessionStart side: `render_checkin_context` must report
    (and heartbeat) the session-resolved worktree, not wherever the hook's
    own process cwd happens to be fixed to."""
    primary, merged_wt, _unmerged_wt = repo_with_worktrees
    board = tmp_path / "agent-board.jsonl"
    session_id = "sess-def-456"

    heartbeat = _run_board_cli(
        merged_wt, board, "heartbeat", "--summary", "prior work", "--session-id", session_id
    )
    assert heartbeat.returncode == 0, heartbeat.stderr
    add = _run_board_cli(
        merged_wt, board, "add", "--kind", "followup", "--severity", "note", "--summary", "remember to check X"
    )
    assert add.returncode == 0, add.stderr

    proc = _run_hook(START_HOOK, primary, board, stdin_payload={"session_id": session_id})
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert "remember to check X" in context


def test_stop_hook_does_not_nag_about_a_different_sessions_items(
    primary_repo: Path, tmp_path: Path
) -> None:
    """Regression test: items are worktree-scoped, not session-scoped, so a
    shared checkout accumulates open items across every past session that
    ever worked there. A fresh session that added zero items of its own
    should not be nagged every Stop about items a *different* session
    opened -- only about its own open items (or legacy items with no
    session_id at all, which have no session to attribute them to)."""
    board = tmp_path / "agent-board.jsonl"
    other_session = "sess-other-111"
    add = _run_board_cli(
        primary_repo,
        board,
        "add",
        "--kind",
        "finding",
        "--severity",
        "note",
        "--summary",
        "found by a different session",
        "--session-id",
        other_session,
    )
    assert add.returncode == 0, add.stderr

    proc = _run_hook(STOP_HOOK, primary_repo, board, stdin_payload={"session_id": "sess-me-222"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_stop_hook_still_nags_about_this_sessions_own_items(
    primary_repo: Path, tmp_path: Path
) -> None:
    """Items this session itself opened must still surface in the nag --
    the fix scopes out *other* sessions' items, not all items."""
    board = tmp_path / "agent-board.jsonl"
    session_id = "sess-me-333"
    add = _run_board_cli(
        primary_repo,
        board,
        "add",
        "--kind",
        "finding",
        "--severity",
        "note",
        "--summary",
        "found by me",
        "--session-id",
        session_id,
    )
    assert add.returncode == 0, add.stderr

    proc = _run_hook(STOP_HOOK, primary_repo, board, stdin_payload={"session_id": session_id})
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "1 open item" in payload["hookSpecificOutput"]["additionalContext"]


def test_stop_hook_still_nags_when_its_own_session_id_is_unresolved(
    primary_repo: Path, tmp_path: Path
) -> None:
    """Regression test for a review-caught bug: if the hook can't resolve
    its OWN session_id (missing/malformed stdin payload, or a non-Claude-Code
    harness whose Stop payload doesn't carry one), it can't tell "this
    session's item" from "some other session's item" either. The fix must
    fail open (fall back to plain worktree scoping) in that case, not
    silently treat every session_id-tagged item as belonging to someone
    else. A naive `item_session_id == session_id` check with session_id=None
    would incorrectly exclude this item."""
    board = tmp_path / "agent-board.jsonl"
    add = _run_board_cli(
        primary_repo,
        board,
        "add",
        "--kind",
        "finding",
        "--severity",
        "note",
        "--summary",
        "tagged item",
        "--session-id",
        "sess-real-555",
    )
    assert add.returncode == 0, add.stderr

    # No stdin_payload at all -- session_id resolves to None, same as a
    # malformed/absent Stop payload.
    proc = _run_hook(STOP_HOOK, primary_repo, board)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "1 open item" in payload["hookSpecificOutput"]["additionalContext"]


def test_stop_hook_still_nags_about_legacy_items_with_no_session_id(
    primary_repo: Path, tmp_path: Path
) -> None:
    """Items added before this field existed (or via a plain shell call
    outside Claude Code) have no session_id to attribute them to -- they
    must keep nagging rather than silently going unowned forever."""
    board = tmp_path / "agent-board.jsonl"
    add = _run_board_cli(
        primary_repo, board, "add", "--kind", "finding", "--severity", "note", "--summary", "legacy item"
    )
    assert add.returncode == 0, add.stderr

    proc = _run_hook(STOP_HOOK, primary_repo, board, stdin_payload={"session_id": "sess-me-444"})
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "1 open item" in payload["hookSpecificOutput"]["additionalContext"]


def test_cursor_hooks_file_calls_agent_board_scripts() -> None:
    hooks_path = ROOT / ".cursor" / "hooks.json"
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    rendered = json.dumps(data)
    assert "sessionStart" in rendered
    assert "stop" in rendered
    assert "session_start_agent_board.py" in rendered
    assert "session_stop_agent_board.py" in rendered
