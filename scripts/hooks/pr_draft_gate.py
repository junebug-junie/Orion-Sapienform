#!/usr/bin/env python3
"""PreToolUse hook (Bash): every pull request opens as a draft and leaves draft only after the session
answered Orion's first review (Juniper 2026-11-25: "claude pushes prs in draft form, and moves it out
of draft form after they respond to Orion's first review. this is mandatory").

Two rules, both refusals (``permissionDecision: deny``), never a silent rewrite:

1. ``gh pr create`` without ``--draft`` / ``-d`` is refused.
2. ``gh pr ready`` (not ``--undo``) is refused until the PR's session thread (Hub,
   ``GET /api/session-threads?pr=N``) shows one of:
   - Orion asked questions, and a session wrote on the thread after Orion's FIRST question turn;
   - Orion's first answer was ``NO QUESTIONS`` (nothing to answer);
   - Orion never answered and the thread is ``failed`` (Hub stopped trying), or is older than
     ``ORION_PR_READY_WAIT_H`` hours (default 24: cabinet heat can hold Orion's turns for days, and a
     PR must not sit in draft for ever because Orion's body was busy).
   No thread for the PR, or Hub down: refused, with the command that sends the PR to Orion.

Conscious override, per command: ``ORION_ALLOW_PR_READY=1 gh pr ready N``. Kill switch for the whole
hook: ``ORION_PR_DRAFT_GATE=0``. Errors in the hook itself never block (exit 0, logged): only the two
rules above refuse. Log: ~/.orion/session_threads_hooks.log.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Split a command line into its simple commands (good enough for a gate: quotes holding these are rare).
_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||[;|\n]")
_GH_PR_RE = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:rtk\s+(?:proxy\s+)?)?gh\s+pr\s+(create|ready)\b(.*)$",
                       re.DOTALL)
# `gh pr create` anywhere in the command (its body is often a multi-line heredoc, so its flags may sit
# lines after it: the draft flag is looked for in everything after it, not in one line).
_CREATE_RE = re.compile(r"(?:^|[;&|(\n])\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:rtk\s+(?:proxy\s+)?)?gh\s+pr\s+create\b")
_DRAFT_FLAG_RE = re.compile(r"(?:^|\s)(?:--draft(?:=(?!false\b)\S*)?|-d)(?=\s|$)", re.IGNORECASE)
_HELP_FLAG_RE = re.compile(r"(?:^|\s)(?:--help|-h)(?=\s|$)")
OVERRIDE = "ORION_ALLOW_PR_READY=1"
DEFAULT_WAIT_H = 24.0


def _enabled() -> bool:
    return os.environ.get("ORION_PR_DRAFT_GATE", "1").strip().lower() not in ("0", "false", "off", "no")


def undrafted_create(command: str) -> bool:
    """True when ``command`` runs ``gh pr create`` with no draft flag after it (and not ``--help``)."""
    for m in _CREATE_RE.finditer(command or ""):
        rest = command[m.end():]
        if not _DRAFT_FLAG_RE.search(rest) and not _HELP_FLAG_RE.search(rest):
            return True
    return False


def gh_pr_commands(command: str) -> list[tuple[str, list[str], str]]:
    """Each ``gh pr create|ready`` in ``command``: (verb, its argument tokens, the whole segment)."""
    out = []
    for seg in _SEGMENT_SPLIT_RE.split(command or ""):
        m = _GH_PR_RE.match(seg)
        if not m:
            continue
        try:
            args = shlex.split(m.group(2))
        except ValueError:
            args = m.group(2).split()
        out.append((m.group(1), args, seg))
    return out


def ready_target(args: list[str]) -> str:
    """The PR a ``gh pr ready`` names (number, URL or branch), or '' for the current branch's PR."""
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in ("-R", "--repo"):
            skip_next = True
            continue
        if not a.startswith("-"):
            return a
    return ""


def _when(text: Any) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def ready_verdict(thread: dict[str, Any] | None, turns: list[dict[str, Any]], *, now: datetime,
                  wait_h: float = DEFAULT_WAIT_H) -> tuple[bool, str]:
    """(allowed, why) for marking a PR ready, from its session thread and that thread's turns."""
    if thread is None:
        return False, "Orion has no session thread for this pull request, so Orion has not reviewed it."
    orion = sorted((t for t in turns if t.get("speaker") == "orion" and not (t.get("meta") or {}).get("requeued")),
                   key=lambda t: int(t.get("seq") or 0))
    asked = [t for t in orion if not ((t.get("meta") or {}).get("no_questions")
                                      or (t.get("meta") or {}).get("closing_note"))]
    if asked:
        first = int(asked[0].get("seq") or 0)
        if any(t.get("speaker") == "session" and int(t.get("seq") or 0) > first for t in turns):
            return True, "the session answered Orion's first review"
        return False, (f"Orion asked questions on thread {thread.get('thread_id')} and no session has answered "
                       "them yet.")
    if orion:
        return True, "Orion read it and had no questions"
    if thread.get("status") == "failed":
        return True, "Hub stopped trying to get Orion's review (thread failed)"
    created = _when(thread.get("created_at"))
    if created is not None and now - created >= timedelta(hours=wait_h):
        return True, f"Orion has not answered in {wait_h:g} h (Orion's body was busy); the wait is over"
    return False, (f"Orion has not reviewed thread {thread.get('thread_id')} yet (status "
                   f"{thread.get('status')}). Orion reads at the next round-up (every 15 min).")


def _deny(reason: str) -> None:
    sys.stdout.write(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))


def check(command: str, *, cwd: str, lib: Any, now: datetime) -> str | None:
    """The refusal text for ``command``, or None when it may run."""
    if undrafted_create(command):
        return ("Pull requests open as drafts (CLAUDE.md, Orion review). Run the same command with --draft. "
                "Mark it ready with `gh pr ready <n>` after you answered Orion's first review.")
    for verb, args, seg in gh_pr_commands(command):
        if verb != "ready":
            continue
        if "--undo" in args or any(a in ("--help", "-h") for a in args):
            continue
        if OVERRIDE in seg or os.environ.get("ORION_ALLOW_PR_READY") == "1":
            lib.log(f"pr_draft_gate override cmd={seg.strip()[:120]!r}")
            continue
        target = ready_target(args)
        pr = lib.pr_info(target, cwd=cwd or lib.REPO_ROOT, timeout=8.0)
        if pr is None:
            return (f"The draft gate could not find the pull request ({target or 'current branch'}) with gh pr view, "
                    f"so it cannot check Orion's review. Name it: gh pr ready <n>. Override: {OVERRIDE} gh pr ready <n>.")
        number = int(pr["number"])
        try:
            threads = list(lib.request("GET", f"/api/session-threads?pr={number}", timeout=4.0).get("threads") or [])
            thread = threads[0] if threads else None
            turns = lib.thread_view(str(thread["thread_id"]))[1] if thread else []
        except Exception as exc:  # noqa: BLE001 -- Hub down: the review cannot be checked
            return (f"Hub did not answer ({type(exc).__name__}), so the draft gate cannot check Orion's review of "
                    f"PR #{number}. Retry when Hub is up. Override: {OVERRIDE} gh pr ready {number}.")
        wait_h = float(os.environ.get("ORION_PR_READY_WAIT_H") or DEFAULT_WAIT_H)
        ok, why = ready_verdict(thread, turns, now=now, wait_h=wait_h)
        lib.log(f"pr_draft_gate ready pr={number} allowed={ok} why={why!r}")
        if not ok:
            send = "" if thread else f" Send it: python3 scripts/orion_session_thread.py send --pr {number}."
            return (f"PR #{number} stays a draft: {why}{send} Answer Orion with "
                    "python3 scripts/orion_session_thread.py reply --thread <id> --pr <n> --body-file <file>, "
                    f"then run gh pr ready again. Override (Juniper's call): {OVERRIDE} gh pr ready {number}.")
    return None


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:  # noqa: BLE001
        return
    if not isinstance(payload, dict) or not _enabled():
        return
    command = str((payload.get("tool_input") or {}).get("command") or "")
    if "gh" not in command or "pr" not in command:
        return   # the cheap exit: almost every Bash call
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import orion_session_thread_lib as lib

    try:
        reason = check(command, cwd=str(payload.get("cwd") or ""), lib=lib, now=datetime.now(timezone.utc))
    except Exception as exc:  # noqa: BLE001 -- a bug in the gate never blocks work
        lib.log(f"pr_draft_gate failed err={type(exc).__name__}: {str(exc)[:200]}")
        return
    if reason:
        _deny(reason)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        pass
