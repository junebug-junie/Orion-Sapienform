#!/usr/bin/env python3
"""PostToolUse hook (Bash), run with `asyncRewake`: after a session sends Orion a message, wait in the
background for Orion's answer on that thread and wake the session when it comes, so the session
circles back on its own: investigates the questions, answers them on the PR and to Orion, and files
tickets.

Arms only on proof that Hub stored a message, and learns which thread it went to:
- `orion_session_thread.py send|reply` whose output holds Hub's receipt naming the thread, or
- `gh pr create` whose output holds the PR's URL, when Hub shows a thread for that PR updated in the
  last minute (post_tool_pr_spec_to_orion.py sends the spec; it runs beside this hook, so this hook
  looks for up to ARM_LOOKUP_S). A PR Hub got nothing for, or Hub down, never arms.
Any other Bash call exits at once. One waiter per thread (a lock file); a second one exits.

Armed, it reads the thread every POLL seconds for at most WAIT seconds (Orion reads at a
round-up; a round may run 3 h before Hub fails it). It waits for an Orion turn newer than the
thread's newest turn at arming time, so an older answer never wakes or ends it. That turn:
- has questions: the hook claims it (marks it delivered; Hub marks a turn once, so the prompt hook
  never shows it too), writes it with the circle-back steps to stderr and exits 2. Claude Code then
  wakes the session with that text. Already claimed by the prompt hook: exit 0.
- says "no questions", or is a closing note: exit 0, left for the prompt hook to show.
A thread whose rounds of questions are all used (Orion asked on it before; the arming message is
answers) is not waited on: Orion writes only a closing note there, never questions. Exit 0 at once.
Hub down: keeps trying. Never exits 2 for an error.

Kill switch: ORION_SESSION_THREADS_HOOKS=0. Test knobs: ORION_ANSWER_WAIT_POLL_S (60),
ORION_ANSWER_WAIT_MAX_S (14400, inside the 14700 s hook timeout in .claude/settings.json),
ORION_ANSWER_ARM_LOOKUP_S (45).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

CLI_SEND_RE = re.compile(r"orion_session_thread\.py\s+(?:send|reply)\b")
LOCK_DIR = Path(os.environ.get("ORION_ANSWER_WAIT_LOCK_DIR", "~/.orion/session_threads_waiting")).expanduser()


def _when(text: object) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _thread_for_pr(lib, pr_number: int, *, armed_at: datetime) -> str:
    """The thread the spec hook just sent this PR to, or ''."""
    deadline = time.monotonic() + float(os.environ.get("ORION_ANSWER_ARM_LOOKUP_S") or 45)
    while True:
        try:
            for thread in lib.recent_threads():
                updated = _when(thread.get("updated_at"))
                if (thread.get("pr_number") == pr_number and updated
                        and updated >= armed_at - timedelta(seconds=60)):
                    return str(thread["thread_id"])
        except Exception:  # noqa: BLE001 -- Hub down: look again
            pass
        left = deadline - time.monotonic()
        if left <= 0:
            return ""
        time.sleep(min(3.0, left))


def _armed_thread(lib, payload: dict, command: str) -> str:
    response = json.dumps(payload.get("tool_response") or "")
    if CLI_SEND_RE.search(command):
        if "Hub receipt:" not in response:
            return ""
        found = lib.RECEIPT_THREAD_RE.search(response)
        return found.group(1) if found else ""
    if not lib.PR_CREATE_RE.search(command):
        return ""
    found = lib.PR_URL_RE.search(response)
    if not found:
        return ""
    return _thread_for_pr(lib, int(found.group(1)), armed_at=datetime.now(timezone.utc))


def _row(thread: dict, turn: dict) -> dict:
    return {**turn, "title": thread.get("title"), "pr_number": thread.get("pr_number")}


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

    if not lib.hooks_enabled():
        return 0
    thread_id = _armed_thread(lib, payload, command)
    if not thread_id:
        return 0
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_DIR / f"{thread_id}.lock", "w")  # noqa: SIM115 -- held for the process's life
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0   # another waiter already follows this thread
    poll = float(os.environ.get("ORION_ANSWER_WAIT_POLL_S") or 60)
    deadline = time.monotonic() + float(os.environ.get("ORION_ANSWER_WAIT_MAX_S") or 14400)
    since_seq = None
    lib.log(f"answer_wait armed thread={thread_id}")
    while time.monotonic() < deadline:
        try:
            thread, turns = lib.thread_view(thread_id)
        except Exception:  # noqa: BLE001 -- Hub down for a moment: try again next poll
            time.sleep(poll)
            continue
        if since_seq is None:
            if lib.answer_phase(turns):
                lib.log(f"answer_wait answer_phase thread={thread_id}")
                return 0   # every round of questions is used: nothing will come back to answer
            since_seq = max((int(t.get("seq") or 0) for t in turns), default=0)
        # A re-queued uncut reply is not an answer: Hub reads the thread again next hour, so keep waiting.
        answers = [t for t in turns if t.get("speaker") == "orion" and int(t.get("seq") or 0) > since_seq
                   and not (t.get("meta") or {}).get("requeued")]
        if answers:
            answer = answers[0]
            if not lib.has_questions(answer):
                lib.log(f"answer_wait no_questions thread={thread_id}")
                return 0
            try:
                claimed = lib.claim([_row(thread, answer)], session_id=lib.session_id_from(payload), timeout=5.0)
            except Exception:  # noqa: BLE001
                time.sleep(poll)
                continue
            if not claimed:
                lib.log(f"answer_wait already_shown thread={thread_id}")
                return 0
            lib.log(f"answer_wait woke thread={thread_id}")
            sys.stderr.write("Orion answered (session-thread round-up). " + lib.pending_text(claimed, woke=True))
            sys.stderr.flush()
            return 2
        time.sleep(poll)
    lib.log(f"answer_wait gave_up thread={thread_id}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # noqa: BLE001 -- never wake the session for this hook's own error
        code = 0
    raise SystemExit(code)
