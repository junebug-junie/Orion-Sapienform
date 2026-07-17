from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_board_lib import (  # noqa: E402
    BoardConfig,
    append_event,
    load_state,
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
