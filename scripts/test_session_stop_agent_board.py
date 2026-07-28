"""Tests for the session_stop_agent_board Stop hook's staleness filtering.

Regression coverage for a live bug found 2026-07-24: ownerless items (no
session_id, e.g. pre-dating that field) nagged every session that ever
stopped in the worktree, forever, since `checkout` closes presence but never
touches item status. See scripts/hooks/session_stop_agent_board.py.

Also covers two sub-bugs found live 2026-07-28 under the same symptom
(same nag repeating every Stop turn): (b) no per-turn dedup -- the hook
re-emitted an identical nag on every Stop even when the item set never
changed -- and (c) "parked" items were treated as still-nagging despite the
hook's own remedy text ("resolve/park them") implying parking should quiet
it.
"""
from __future__ import annotations

import importlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/hooks")

hook = importlib.import_module("session_stop_agent_board")


def _state_with_items(items: dict) -> SimpleNamespace:
    return SimpleNamespace(items=items)


def _run_hook(items: dict, session_id: str | None = "current-session", nag_state_dir: Path | None = None) -> str | None:
    with mock.patch.object(hook, "read_session_id_from_stdin_hook_payload", return_value=session_id), \
         mock.patch.object(hook, "board_config_from_env", return_value=object()), \
         mock.patch.object(hook, "resolve_current_identity", return_value={"worktree_path": "/repo"}), \
         mock.patch.object(hook, "load_state", return_value=_state_with_items(items)):
        if nag_state_dir is not None:
            with mock.patch.object(hook, "_nag_state_dir", return_value=nag_state_dir):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    hook.main()
                return buf.getvalue()
        # No isolated dir given: point at a throwaway tmp dir so tests never
        # touch the real ~/.orion/stop_hook_nag_state.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hook, "_nag_state_dir", return_value=Path(tmp)):
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


def test_parked_item_does_not_nag() -> None:
    """Mode (c), found live 2026-07-28: the nag's own remedy text says
    'resolve/park them', implying parking is a valid way to quiet it, but
    the old status filter (`{"open", "parked"}`) kept nagging on parked
    items forever. Parking is a real triage decision, not neglect -- once
    parked, the Stop hook should stop treating it as an unactioned item.
    """
    items = {
        "item-1": {
            "worktree_path": "/repo",
            "status": "parked",
            "session_id": "current-session",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    out = _run_hook(items, session_id="current-session")
    assert out == "", f"expected silence for a parked item, got: {out!r}"


def test_resolved_and_handed_off_items_do_not_nag() -> None:
    items = {
        "item-1": {"worktree_path": "/repo", "status": "resolved", "session_id": "current-session"},
        "item-2": {"worktree_path": "/repo", "status": "handed-off", "session_id": "current-session"},
    }
    out = _run_hook(items, session_id="current-session")
    assert out == "", f"expected silence for resolved/handed-off items, got: {out!r}"


def test_same_item_set_does_not_renag_within_a_session() -> None:
    """Mode (b), found live 2026-07-28 (7+ consecutive Stop turns, zero new
    user input, identical nag every time): no per-turn dedup existed at
    all. The exact same open-item set should only nag once per session,
    not on every single Stop turn.
    """
    items = {
        "item-1": {"worktree_path": "/repo", "status": "open", "session_id": "current-session"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        first = _run_hook(items, session_id="current-session", nag_state_dir=state_dir)
        second = _run_hook(items, session_id="current-session", nag_state_dir=state_dir)
    assert first, "expected the first Stop turn to nag"
    assert second == "", f"expected the second Stop turn (same item set) to stay silent, got: {second!r}"


def test_changed_item_set_renags_after_suppression() -> None:
    items_v1 = {
        "item-1": {"worktree_path": "/repo", "status": "open", "session_id": "current-session"},
    }
    items_v2 = {
        "item-1": {"worktree_path": "/repo", "status": "open", "session_id": "current-session"},
        "item-2": {"worktree_path": "/repo", "status": "open", "session_id": "current-session"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        first = _run_hook(items_v1, session_id="current-session", nag_state_dir=state_dir)
        second = _run_hook(items_v1, session_id="current-session", nag_state_dir=state_dir)
        third = _run_hook(items_v2, session_id="current-session", nag_state_dir=state_dir)
    assert first, "expected the first Stop turn to nag"
    assert second == "", "expected the unchanged-set second turn to stay silent"
    assert third, "expected a new item added to the set to re-trigger the nag"


def test_malicious_session_id_does_not_escape_nag_state_dir() -> None:
    """`session_id` is attacker-influenceable (arbitrary JSON on stdin, per
    `read_session_id_from_stdin_hook_payload`'s own docstring covering
    non-Claude-Code callers) -- a raw `../../etc/whatever` must not let the
    marker file land outside `_nag_state_dir()`.
    """
    items = {
        "item-1": {
            "worktree_path": "/repo",
            "status": "open",
            "session_id": "../../../../tmp/escaped",
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        _run_hook(items, session_id="../../../../tmp/escaped", nag_state_dir=state_dir)
        written = list(state_dir.iterdir())
    assert written, "expected a marker file to be written inside the state dir"
    for path in written:
        assert path.parent == state_dir, f"marker escaped the state dir: {path}"
        assert ".." not in path.name and "/" not in path.name


def test_different_sessions_each_get_their_own_nag() -> None:
    items = {
        "item-1": {"worktree_path": "/repo", "status": "open", "session_id": None,
                   "updated_at": datetime.now(timezone.utc).isoformat()},
    }
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        session_a_first = _run_hook(items, session_id="session-a", nag_state_dir=state_dir)
        session_a_second = _run_hook(items, session_id="session-a", nag_state_dir=state_dir)
        session_b_first = _run_hook(items, session_id="session-b", nag_state_dir=state_dir)
    assert session_a_first, "expected session-a's first Stop turn to nag"
    assert session_a_second == "", "expected session-a's second Stop turn (same set) to stay silent"
    assert session_b_first, "expected session-b to get its own fresh nag, not inherit session-a's suppression"


if __name__ == "__main__":
    test_stale_ownerless_item_does_not_nag()
    test_fresh_ownerless_item_still_nags()
    test_own_session_item_nags_regardless_of_age()
    test_mixed_stale_and_fresh_ownerless_items_counts_only_fresh()
    test_unresolved_session_id_fails_open_ignoring_staleness()
    test_parked_item_does_not_nag()
    test_resolved_and_handed_off_items_do_not_nag()
    test_same_item_set_does_not_renag_within_a_session()
    test_changed_item_set_renags_after_suppression()
    test_malicious_session_id_does_not_escape_nag_state_dir()
    test_different_sessions_each_get_their_own_nag()
    print("all tests passed")
