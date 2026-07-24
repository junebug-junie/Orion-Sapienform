#!/usr/bin/env python3
"""Stop hook: reminds the agent to checkout or resolve open board items.

Fails silently (no output) on any error -- never blocks session stop.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import (  # noqa: E402
    board_config_from_env,
    load_state,
    read_session_id_from_stdin_hook_payload,
    resolve_current_identity,
)

# Ownerless items (no session_id) older than this stop nagging every session
# that ever stops in the worktree -- see _is_stale_ownerless_item below.
_OWNERLESS_ITEM_STALE_AFTER = timedelta(hours=24)


def _is_stale_ownerless_item(item: dict) -> bool:
    raw = item.get("updated_at")
    if not raw:
        # No timestamp to judge age by -- keep the old fail-open behavior
        # (treat as fresh, i.e. still nag) rather than guessing.
        return False
    try:
        updated_at = datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated_at) > _OWNERLESS_ITEM_STALE_AFTER


def main() -> None:
    if os.environ.get("ORION_FCC_SUBPROCESS"):
        # See matching guard in session_start_agent_board.py: this is a
        # per-turn FCC chat subprocess, not a coding session with a worktree
        # to check out of.
        return
    session_id = read_session_id_from_stdin_hook_payload()
    try:
        cfg = board_config_from_env()
        current = resolve_current_identity(cfg, session_id=session_id)["worktree_path"]
        state = load_state(cfg)
        # Items are worktree-scoped, not session-scoped, so a shared checkout
        # can accumulate open items across many past sessions -- without the
        # session_id check below, a fresh read-only session that added zero
        # items still gets nagged every Stop about work it never touched.
        # Items with no session_id (added before this field existed, or by a
        # plain shell call outside Claude Code) have no session to attribute
        # them to, so a *recent* one still nags rather than going silently
        # unowned. But without a staleness cutoff this becomes permanent: a
        # legacy ownerless item never stops nagging, for every future session
        # that stops in the worktree, since `checkout` closes presence, not
        # items -- live-confirmed 2026-07-24 (dozens of pre-session_id-field
        # findings nagging a session that never touched them). Past the
        # cutoff, treat it as backlog to triage explicitly, not a per-Stop
        # reminder. If *our own* session_id couldn't be resolved
        # (missing/malformed stdin payload, or a non-Claude-Code harness
        # whose Stop payload doesn't carry one), we can't tell "mine" from
        # "someone else's" either -- fail open (fall back to plain worktree
        # scoping, no staleness filter) rather than silently excluding every
        # item that happens to carry a real session_id.
        open_items = [
            item for item in state.items.values()
            if item.get("worktree_path") == current
            and item.get("status") in {"open", "parked"}
            and (
                session_id is None
                or item.get("session_id") == session_id
                or (not item.get("session_id") and not _is_stale_ownerless_item(item))
            )
        ]
    except Exception:
        return
    if not open_items:
        # Nothing actionable -- saying so on every single Stop is pure token
        # noise, not a reminder. Stay silent, matching the fail-silent
        # pattern already used for errors above.
        return
    detail = f"Agent board checkout reminder: {len(open_items)} open item(s) remain for this worktree. Run `python3 scripts/agent_board.py checkout` or resolve/park them."
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": detail,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
