#!/usr/bin/env python3
"""UserPromptSubmit / SessionStart hook: hand Orion's questions back to the coding session that
sent the spec.

Asks Hub for Orion's turns not yet shown to this session, for this session id OR this session's branch
(the agent board's identity: a hook's own cwd is not reliable), claims them for this session (only the
rows this hook claims, so the waiting hook never shows the same ones), then prints them as
additionalContext. Hub keeps a question turn pending for later sessions until one replies on the thread.

Silent and fast when nothing is pending or Hub is down (2 s timeout, exit 0, no output).
Kill switch: ORION_SESSION_THREADS_HOOKS=0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:  # noqa: BLE001
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import orion_session_thread_lib as lib

    if not lib.hooks_enabled():
        return
    try:
        session_id = lib.session_id_from(payload)
        branch = lib.identity(session_id)["branch"]
        if branch in ("main", "master", "HEAD"):
            branch = ""   # the shared checkout's branch is not one session's
        if session_id == "manual" and not branch:
            return
        rows = lib.pending(session_id=session_id, branch=branch, timeout=2.0)
    except Exception:  # noqa: BLE001 -- Hub down or an older Hub: silent
        return
    try:
        rows = lib.claim(rows, session_id=session_id, timeout=2.0)   # claim first: a waiting hook may show them
    except Exception:  # noqa: BLE001
        return
    text = lib.pending_text(rows)
    if not text:
        return
    event = str(payload.get("hook_event_name") or "UserPromptSubmit")
    sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}},
                                ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 -- never a non-zero exit from this hook
        pass
