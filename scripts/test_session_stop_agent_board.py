"""Tests for the session_stop_agent_board Stop hook's staleness filtering.

Regression coverage for a live bug found 2026-07-24: ownerless items (no
session_id, e.g. pre-dating that field) nagged every session that ever
stopped in the worktree, forever, since `checkout` closes presence but never
touches item status. See scripts/hooks/session_stop_agent_board.py.
"""
from __future__ import annotations

import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/hooks")

hook = importlib.import_module("session_stop_agent_board")


def _state_with_items(items: dict) -> SimpleNamespace:
    return SimpleNamespace(items=items)


def _run_hook(items: dict, session_id: str | None = "current-session") -> str | None:
    with mock.patch.object(hook, "read_session_id_from_stdin_hook_payload", return_value=session_id), \
         mock.patch.object(hook, "board_config_from_env", return_value=object()), \
         mock.patch.object(hook, "resolve_current_identity", return_value={"worktree_path": "/repo"}), \
         mock.patch.object(hook, "load_state", return_value=_state_with_items(items)):
        buf = io.StringIO()
        with redirect_stdout(buf):
            hook.main()
        return buf.getvalue()


def test_stale_ownerless_item_does_not_nag() -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    items = {
        "item-1": {
            "worktree_path": "/repo",
            "status": "open",
            "session_id": None,
            "updated_at": old_ts,
        }
    }
    out = _run_hook(items)
    assert out == "", f"expected silence for stale ownerless item, got: {out!r}"


def test_fresh_ownerless_item_still_nags() -> None:
    fresh_ts = datetime.now(timezone.utc).isoformat()
    items = {
        "item-1": {
            "worktree_path": "/repo",
            "status": "open",
            "session_id": None,
            "updated_at": fresh_ts,
        }
    }
    out = _run_hook(items)
    assert out, "expected a nag for a fresh ownerless item"
    payload = json.loads(out)
    assert "1 open item" in payload["hookSpecificOutput"]["additionalContext"]


def test_own_session_item_nags_regardless_of_age() -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    items = {
        "item-1": {
            "worktree_path": "/repo",
            "status": "open",
            "session_id": "current-session",
            "updated_at": old_ts,
        }
    }
    out = _run_hook(items, session_id="current-session")
    assert out, "expected a nag for the current session's own old item"


def test_mixed_stale_and_fresh_ownerless_items_counts_only_fresh() -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    items = {
        "stale-1": {"worktree_path": "/repo", "status": "open", "session_id": None, "updated_at": old_ts},
        "stale-2": {"worktree_path": "/repo", "status": "open", "session_id": None, "updated_at": old_ts},
        "fresh-1": {"worktree_path": "/repo", "status": "open", "session_id": None, "updated_at": fresh_ts},
    }
    out = _run_hook(items)
    assert out, "expected a nag since one item is fresh"
    payload = json.loads(out)
    assert "1 open item" in payload["hookSpecificOutput"]["additionalContext"], payload


def test_unresolved_session_id_fails_open_ignoring_staleness() -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    items = {
        "item-1": {
            "worktree_path": "/repo",
            "status": "open",
            "session_id": None,
            "updated_at": old_ts,
        }
    }
    out = _run_hook(items, session_id=None)
    assert out, "expected fail-open nag when our own session_id can't be resolved"


if __name__ == "__main__":
    test_stale_ownerless_item_does_not_nag()
    test_fresh_ownerless_item_still_nags()
    test_own_session_item_nags_regardless_of_age()
    test_mixed_stale_and_fresh_ownerless_items_counts_only_fresh()
    test_unresolved_session_id_fails_open_ignoring_staleness()
    print("all tests passed")
