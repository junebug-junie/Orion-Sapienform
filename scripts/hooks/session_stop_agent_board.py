#!/usr/bin/env python3
"""Stop hook: reminds the agent to checkout or resolve open board items.

Fails silently (no output) on any error -- never blocks session stop.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import (  # noqa: E402
    board_config_from_env,
    load_state,
    read_session_id_from_stdin_hook_payload,
    resolve_current_identity,
)


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
        # them to, so they keep nagging rather than going silently unowned.
        # If *our own* session_id couldn't be resolved (missing/malformed
        # stdin payload, or a non-Claude-Code harness whose Stop payload
        # doesn't carry one), we can't tell "mine" from "someone else's"
        # either -- fail open (fall back to plain worktree scoping) rather
        # than silently excluding every item that happens to carry a real
        # session_id.
        open_items = [
            item for item in state.items.values()
            if item.get("worktree_path") == current
            and item.get("status") in {"open", "parked"}
            and (session_id is None or not item.get("session_id") or item.get("session_id") == session_id)
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
