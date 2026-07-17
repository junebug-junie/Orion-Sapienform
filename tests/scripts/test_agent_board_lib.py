from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_board_lib import (  # noqa: E402
    BoardConfig,
    append_event,
    detect_collisions,
    load_state,
    reconcile_closed_worktrees,
    render_checkin_context,
    resolve_current_identity,
    validate_item_payload,
)


def _config(tmp_path: Path) -> BoardConfig:
    return BoardConfig(board_path=tmp_path / "agent-board.jsonl", stale_after_minutes=30)


def test_append_event_creates_parent_dir_and_valid_jsonl(tmp_path: Path) -> None:
    cfg = _config(tmp_path / "nested")
    event = append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
        now=datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
    )

    assert cfg.board_path.exists()
    raw = cfg.board_path.read_text(encoding="utf-8").strip()
    decoded = json.loads(raw)
    assert decoded["type"] == "presence_upserted"
    assert decoded["at"] == "2026-07-16T12:00:00+00:00"
    assert decoded["payload"]["worktree_path"] == "/repo/wt-a"
    assert event.type == "presence_upserted"


def test_load_state_materializes_last_presence_event(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(cfg, "presence_upserted", {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"})
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/wt-a",
            "branch": "feat/a",
            "status": "active",
            "thread_summary": "Working on the board.",
            "current_task": "Writing tests.",
        },
    )

    state = load_state(cfg, live_worktrees={"/repo/wt-a"})

    assert state.presence["/repo/wt-a"]["thread_summary"] == "Working on the board."
    assert state.presence["/repo/wt-a"]["current_task"] == "Writing tests."


def test_owner_scope_requires_scope_note_when_not_this_worktree() -> None:
    with pytest.raises(ValueError, match="scope_note is required"):
        validate_item_payload(
            {
                "id": "item-1",
                "kind": "finding",
                "severity": "should",
                "owner_scope": "juniper",
                "worktree_path": "/repo/wt-a",
                "summary": "Needs a scope note.",
                "status": "open",
            }
        )


def test_item_validation_accepts_this_worktree_without_scope_note() -> None:
    validate_item_payload(
        {
            "id": "item-1",
            "kind": "finding",
            "severity": "should",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/wt-a",
            "summary": "Scoped to current worktree.",
            "status": "open",
        }
    )


def test_old_active_presence_becomes_stale(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
        now=datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
    )

    state = load_state(
        cfg,
        now=datetime(2026, 7, 16, 12, 31, tzinfo=UTC),
        live_worktrees={"/repo/wt-a"},
    )

    assert state.presence["/repo/wt-a"]["status"] == "stale"


def test_missing_live_worktree_becomes_closed(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
    )

    state = load_state(cfg, live_worktrees={"/repo/other"})

    assert state.presence["/repo/wt-a"]["status"] == "closed"


def test_concurrent_appends_do_not_corrupt_jsonl(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    def write_one(index: int) -> None:
        append_event(
            cfg,
            "item_upserted",
            {
                "id": f"item-{index}",
                "kind": "finding",
                "severity": "note",
                "owner_scope": "this-worktree",
                "worktree_path": "/repo/wt-a",
                "summary": f"Finding {index}",
                "status": "open",
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_one, range(50)))

    lines = cfg.board_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50
    decoded = [json.loads(line) for line in lines]
    assert {row["payload"]["id"] for row in decoded} == {f"item-{index}" for index in range(50)}


def test_append_event_creates_private_board_file_and_directory(tmp_path: Path) -> None:
    cfg = _config(tmp_path / "nested")

    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
    )

    assert oct(cfg.board_path.parent.stat().st_mode & 0o777) == "0o700"
    assert oct(cfg.board_path.stat().st_mode & 0o777) == "0o600"


def test_append_event_tolerates_unwritable_shared_parent() -> None:
    """Board files may live under shared parents like /tmp for smoke isolation.
    chmod on that parent must fail soft rather than break board writes."""
    shared = Path("/tmp")
    board = shared / f"orion-agent-board-test-{os.getpid()}.jsonl"
    lock = Path(str(board) + ".lock")
    board.unlink(missing_ok=True)
    lock.unlink(missing_ok=True)
    cfg = BoardConfig(board_path=board, stale_after_minutes=30)

    try:
        append_event(
            cfg,
            "presence_upserted",
            {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
        )
        assert board.exists()
        assert oct(board.stat().st_mode & 0o777) == "0o600"
    finally:
        board.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def test_load_state_skips_corrupt_jsonl_lines(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
    )
    with cfg.board_path.open("a", encoding="utf-8") as handle:
        handle.write("{bad\n")
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/wt-b",
            "branch": "feat/b",
            "status": "active",
        },
    )

    state = load_state(cfg, live_worktrees={"/repo/wt-a", "/repo/wt-b"})

    assert set(state.presence) == {"/repo/wt-a", "/repo/wt-b"}


def test_render_checkin_includes_this_worktree_global_strip_and_presence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/current",
            "branch": "feat/current",
            "status": "active",
            "thread_summary": "Current thread.",
            "current_task": "Implementing checkin.",
        },
    )
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/other",
            "branch": "feat/other",
            "status": "active",
            "thread_summary": "Other agent summary.",
            "current_task": "Touching scripts.",
        },
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "current-item",
            "kind": "finding",
            "severity": "should",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/current",
            "summary": "Current worktree finding.",
            "status": "open",
            "related_files": ["scripts/agent_board.py"],
        },
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "global-blocker",
            "kind": "blocker",
            "severity": "blocker",
            "owner_scope": "juniper",
            "scope_note": "Needs Juniper decision.",
            "worktree_path": "/repo/other",
            "summary": "Global blocker.",
            "status": "open",
            "related_files": [],
        },
    )
    monkeypatch.setattr(
        "agent_board_lib.current_worktree_identity",
        lambda: {"worktree_path": "/repo/current", "branch": "feat/current"},
    )
    monkeypatch.setattr("agent_board_lib.live_worktree_paths", lambda: {"/repo/current", "/repo/other"})

    output = render_checkin_context(cfg)

    assert "This worktree" in output
    assert "Current worktree finding." in output
    assert "Global strip" in output
    assert "Global blocker." in output
    assert "Workspace presence" in output
    assert "Other agent summary." in output


def test_detect_collisions_reports_overlapping_related_files(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/current", "branch": "feat/current", "status": "active"},
    )
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/other", "branch": "feat/other", "status": "active"},
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "current-item",
            "kind": "finding",
            "severity": "note",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/current",
            "summary": "Current file touch.",
            "status": "open",
            "related_files": ["scripts/agent_board.py"],
        },
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "other-item",
            "kind": "finding",
            "severity": "note",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/other",
            "summary": "Other file touch.",
            "status": "open",
            "related_files": ["scripts/agent_board.py"],
        },
    )
    state = load_state(cfg, live_worktrees={"/repo/current", "/repo/other"})

    collisions = detect_collisions(state, "/repo/current")

    assert collisions == [
        "Potential collision with /repo/other: overlapping files scripts/agent_board.py"
    ]


def test_detect_collisions_reports_same_service_path(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/current", "branch": "feat/current", "status": "active"},
    )
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/other", "branch": "feat/other", "status": "active"},
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "current-item",
            "kind": "finding",
            "severity": "note",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/current",
            "summary": "Current service touch.",
            "status": "open",
            "related_files": ["services/orion-hub/app/main.py"],
        },
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "other-item",
            "kind": "finding",
            "severity": "note",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/other",
            "summary": "Other service touch.",
            "status": "open",
            "related_files": ["services/orion-hub/tests/test_main.py"],
        },
    )
    state = load_state(cfg, live_worktrees={"/repo/current", "/repo/other"})

    collisions = detect_collisions(state, "/repo/current")

    assert collisions == [
        "Potential collision with /repo/other: shared service paths services/orion-hub"
    ]


def test_reconcile_closed_worktrees_persists_closed_event(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/old", "branch": "feat/old", "status": "active"},
    )

    state = reconcile_closed_worktrees(cfg, live_paths={"/repo/current"})

    lines = cfg.board_path.read_text(encoding="utf-8").splitlines()
    closed_events = [
        json.loads(line) for line in lines if json.loads(line)["type"] == "presence_closed"
    ]
    assert any(
        event["payload"]["worktree_path"] == "/repo/old" for event in closed_events
    )

    persisted = load_state(cfg)
    assert persisted.presence["/repo/old"]["status"] == "closed"
    assert state.presence["/repo/old"]["status"] == "closed"


def test_resolve_current_identity_finds_worktree_by_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core regression coverage for the Claude-Code hook-cwd bug: a Stop/
    SessionStart hook's own process cwd is fixed to wherever the session
    originally started and does not track mid-session `cd` calls, so
    resolving identity via git-rev-parse-from-cwd resolves to the wrong
    worktree. A git-hook-driven heartbeat (which DOES run with correct cwd)
    tags its presence row with a session_id; resolve_current_identity must
    prefer that over the ambient cwd when a session_id is given."""
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/real-worktree", "branch": "feat/real", "session_id": "sess-1"},
    )
    monkeypatch.setattr(
        "agent_board_lib.current_worktree_identity",
        lambda: {"worktree_path": "/repo/wrong-fixed-cwd", "branch": "main"},
    )

    identity = resolve_current_identity(cfg, session_id="sess-1")

    assert identity["worktree_path"] == "/repo/real-worktree"
    assert identity["branch"] == "feat/real"


def test_resolve_current_identity_picks_most_recent_when_session_touched_multiple_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same session_id can legitimately appear on multiple worktrees'
    presence rows over a session's lifetime (an agent working across several
    worktrees in one conversation) -- resolution must pick the most
    recently heartbeated one, not just any match."""
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/earlier-worktree",
            "branch": "feat/earlier",
            "session_id": "sess-2",
            "heartbeat_at": "2026-07-17T10:00:00+00:00",
        },
    )
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/later-worktree",
            "branch": "feat/later",
            "session_id": "sess-2",
            "heartbeat_at": "2026-07-17T12:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        "agent_board_lib.current_worktree_identity",
        lambda: {"worktree_path": "/repo/wrong-fixed-cwd", "branch": "main"},
    )

    identity = resolve_current_identity(cfg, session_id="sess-2")

    assert identity["worktree_path"] == "/repo/later-worktree"


def test_resolve_current_identity_falls_back_to_cwd_when_no_session_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        "agent_board_lib.current_worktree_identity",
        lambda: {"worktree_path": "/repo/fallback", "branch": "main"},
    )

    identity = resolve_current_identity(cfg, session_id="no-such-session")

    assert identity["worktree_path"] == "/repo/fallback"


def test_resolve_current_identity_falls_back_when_session_id_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        "agent_board_lib.current_worktree_identity",
        lambda: {"worktree_path": "/repo/fallback", "branch": "main"},
    )

    identity = resolve_current_identity(cfg, session_id=None)

    assert identity["worktree_path"] == "/repo/fallback"
