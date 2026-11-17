#!/usr/bin/env python3
"""PostToolUse hook (Bash), run with `asyncRewake`: after a session sends Orion a spec, wait in the
background for Orion's answer and wake the session when it comes, so the session circles back on its
own: investigates the questions, answers them on the PR and to Orion, and files tickets.

Arms only when a message to Orion was really sent:
- `gh pr create` whose output holds the PR's URL, and the PR carries a docs/superpowers/specs/*.md
  file (the same rule post_tool_pr_spec_to_orion.py sends by), or
- `orion_session_thread.py send` whose output holds Hub's receipt.
Any other Bash call exits at once.

Armed, it asks Hub for this session's (or branch's) undelivered answers every POLL seconds, for at
most WAIT seconds (Orion reads at an hourly round-up). When an answer with questions arrives it marks
the answers delivered, writes them with the circle-back steps to stderr and exits 2: Claude Code then
wakes the session with that text. "No questions" alone ends the wait quietly (exit 0) and is left for
the prompt hook to show. Hub down: keeps trying until WAIT runs out. Never exits 2 for an error.

Kill switch: ORION_SESSION_THREADS_HOOKS=0. Test knobs: ORION_ANSWER_WAIT_POLL_S (60),
ORION_ANSWER_WAIT_MAX_S (14400: a round may run 3 h before Hub fails it; inside the 14700 s hook timeout in .claude/settings.json).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

SEND_CLI_RE = re.compile(r"orion_session_thread\.py\s+send\b")


def _armed(lib, payload: dict, command: str) -> bool:
    response = json.dumps(payload.get("tool_response") or "")
    if SEND_CLI_RE.search(command):
        return "Hub receipt:" in response
    if not lib.PR_CREATE_RE.search(command):
        return False
    found = lib.PR_URL_RE.search(response)
    if not found:
        return False
    pr = lib.pr_info(found.group(1), timeout=8.0)
    return bool(pr and lib.spec_paths(pr))


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:  # noqa: BLE001
        return 0
    if not isinstance(payload, dict):
        return 0
    command = str((payload.get("tool_input") or {}).get("command") or "")
    if "gh" not in command and "orion_session_thread" not in command:
        return 0   # the cheap exit: almost every Bash call
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import orion_session_thread_lib as lib

    if not lib.hooks_enabled() or not _armed(lib, payload, command):
        return 0
    session_id = lib.session_id_from(payload)
    branch = lib.identity(session_id)["branch"]
    if branch in ("main", "master", "HEAD"):
        branch = ""
    poll = float(os.environ.get("ORION_ANSWER_WAIT_POLL_S") or 60)
    deadline = time.monotonic() + float(os.environ.get("ORION_ANSWER_WAIT_MAX_S") or 14400)
    lib.log(f"answer_wait armed session={session_id} branch={branch}")
    while time.monotonic() < deadline:
        time.sleep(poll)
        try:
            rows = lib.pending(session_id=session_id, branch=branch, timeout=5.0)
        except Exception:  # noqa: BLE001 -- Hub down for a moment: try again next poll
            continue
        if not rows:
            continue
        if not any(lib.has_questions(r) for r in rows):
            lib.log(f"answer_wait no_questions session={session_id}")
            return 0
        try:
            lib.delivered([str(r["turn_id"]) for r in rows], timeout=5.0)
        except Exception as exc:  # noqa: BLE001 -- the prompt hook may show them again; better twice than never
            lib.log(f"answer_wait delivered_failed err={type(exc).__name__}")
        lib.log(f"answer_wait woke session={session_id} turns={len(rows)}")
        sys.stderr.write("Orion answered (hourly round-up). " + lib.pending_text(rows))
        sys.stderr.flush()
        return 2
    lib.log(f"answer_wait gave_up session={session_id}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # noqa: BLE001 -- never wake the session for this hook's own error
        code = 0
    raise SystemExit(code)
