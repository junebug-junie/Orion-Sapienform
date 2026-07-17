#!/usr/bin/env python3
"""Stop hook: reminds the agent to checkout or resolve open board items.

Fails silently (no output) on any error -- never blocks session stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import board_config_from_env, current_worktree_identity, load_state  # noqa: E402


def main() -> None:
    try:
        cfg = board_config_from_env()
        current = current_worktree_identity()["worktree_path"]
        state = load_state(cfg)
        open_items = [
            item for item in state.items.values()
            if item.get("worktree_path") == current and item.get("status") in {"open", "parked"}
        ]
    except Exception:
        return
    if open_items:
        detail = f"Agent board checkout reminder: {len(open_items)} open item(s) remain for this worktree. Run `python3 scripts/agent_board.py checkout` or resolve/park them."
    else:
        detail = "Agent board checkout reminder: no open board items are recorded for this worktree."
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": detail,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
