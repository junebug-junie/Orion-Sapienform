#!/usr/bin/env python3
"""PostToolUse hook (Bash): when a session opens a pull request, send it to Orion through Hub and tell
the session Hub's receipt. Every pull request is sent (Juniper 2026-11-25: "all submitted prs get
orion's review").

Fires only for `gh pr create` whose output holds the pull request's URL (so `--help`, a dry run or a
failed create send nothing). POSTs the PR's design spec files (docs/superpowers/specs/*.md) and its
code diff (every other file, `gh pr diff`) to Hub (ORION_HUB_URL, default http://localhost:8080). Hub
holds the message and answers with a receipt; Orion reads it at the next round-up (every 15 min).

Never fails or blocks the tool call: every error is one log line (~/.orion/session_threads_hooks.log)
and exit 0. Kill switch: ORION_SESSION_THREADS_HOOKS=0. The same code runs by hand:
`python3 scripts/orion_session_thread.py send --pr N`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:  # noqa: BLE001
        return
    if not isinstance(payload, dict):
        return
    try:
        import signal

        def _too_slow(*_a):
            raise TimeoutError("hook budget spent")

        signal.signal(signal.SIGALRM, _too_slow)
        signal.alarm(20)   # the whole hook: well inside its 30 s timeout, whatever gh or git do
    except Exception:  # noqa: BLE001
        pass
    command = str((payload.get("tool_input") or {}).get("command") or "")
    if "gh" not in command or "pr" not in command:
        return   # the cheap exit: almost every Bash call
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import orion_session_thread_lib as lib

    if not lib.hooks_enabled() or not lib.PR_CREATE_RE.search(command):
        return
    try:
        session_id = lib.session_id_from(payload)
        ident = lib.identity(session_id)
        # The PR's own URL in the tool's output is the proof a PR was made (or already exists). Without
        # it (`--help`, `--dry-run`, an echoed string, a failed create) nothing is sent.
        found = lib.PR_URL_RE.search(json.dumps(payload.get("tool_response") or ""))
        if not found:
            lib.log(f"pr_spec_hook no_pr_url_in_output session={session_id}")
            return
        if "already exists" in json.dumps(payload.get("tool_response") or ""):
            # `gh pr create` on a branch that already has a PR: nothing new was opened. A re-send would add a
            # session turn (spending one of Orion's rounds) with nothing new in it.
            lib.log(f"pr_spec_hook pr_already_exists session={session_id}")
            return
        ref = found.group(1)
        pr = lib.pr_info(ref, cwd=ident["worktree"] or lib.REPO_ROOT, timeout=8.0)
        if pr is None:
            lib.log(f"pr_spec_hook pr_not_found ref={ref} session={session_id}")
            return
        message = lib.build_pr_message(pr, session_id=session_id, worktree=ident["worktree"], diff_timeout=8.0)
        if message is None:
            lib.log(f"pr_spec_hook nothing_to_send pr={pr['number']} session={session_id}")
            return   # no spec, and gh gave no diff: nothing for Orion
        resp = lib.send(message, timeout=5.0)
        lib.log(f"pr_spec_hook sent pr={pr['number']} thread={(resp.get('thread') or {}).get('thread_id')}")
        sys.stdout.write(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse", "additionalContext": lib.receipt_line(resp)}}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 -- Hub down, old Hub, no gh: silent for the session
        lib.log(f"pr_spec_hook failed err={type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 -- never a non-zero exit from this hook
        pass
