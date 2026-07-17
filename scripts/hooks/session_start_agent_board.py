#!/usr/bin/env python3
"""SessionStart hook: surfaces agent workspace board checkin context.

Fails silently (no output) on any error -- never blocks session start.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import (  # noqa: E402
    board_config_from_env,
    read_session_id_from_stdin_hook_payload,
    render_checkin_context,
)


def main() -> None:
    if os.environ.get("ORION_FCC_SUBPROCESS"):
        # This is a per-turn `claude -p` call spawned by orion/harness/fcc_motor.py,
        # not a human/agent coding session -- it has no worktree of its own to
        # check in/out of. Without this, every Orion chat turn writes a
        # spurious presence row to the shared board (see AGENTS.md's agent
        # workspace board section) and inflates the FCC harness step count.
        return
    session_id = read_session_id_from_stdin_hook_payload()
    try:
        context = render_checkin_context(board_config_from_env(), session_id=session_id)
    except Exception:
        return
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
