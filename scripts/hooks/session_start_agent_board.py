#!/usr/bin/env python3
"""SessionStart hook: surfaces agent workspace board checkin context.

Fails silently (no output) on any error -- never blocks session start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import board_config_from_env, render_checkin_context  # noqa: E402


def main() -> None:
    try:
        context = render_checkin_context(board_config_from_env())
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
