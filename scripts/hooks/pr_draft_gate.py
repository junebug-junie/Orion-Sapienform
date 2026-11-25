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
   No thread for the PR (the create hook never reached Hub): the gate sends the PR to Orion itself and
   refuses, so the review and the wait clock start now. Hub down: refused, with the override named.
   Rounds 2 and 3 are not waited on: Juniper's rule is "after they respond to Orion's first review".

The override is reported on the PR's thread as a session turn, so Orion sees each waived review.

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

OVERRIDE = "ORION_ALLOW_PR_READY=1"
DEFAULT_WAIT_H = 24.0
_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "\n", ";;", "|&"}
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _enabled() -> bool:
    return os.environ.get("ORION_PR_DRAFT_GATE", "1").strip().lower() not in ("0", "false", "off", "no")


def strip_heredocs(command: str) -> str:
    """``command`` without heredoc bodies: their lines are text, never commands or flags."""
    out, pending = [], []
    for line in (command or "").split("\n"):
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        out.append(line)
        pending.extend(m.group(2) for m in _HEREDOC_RE.finditer(line))
    return "\n".join(out)


def simple_commands(command: str) -> list[list[str]]:
    """``command`` as its simple commands, each a list of shell words. Quoted text is one word, so text
    inside a PR body or a commit message never reads as a command or a flag. Raises ValueError on
    unbalanced quotes."""
    lex = shlex.shlex(strip_heredocs(command), posix=True, punctuation_chars=";&|()\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    out, cur = [], []
    for tok in lex:
        if tok in _SEPARATORS or (tok and set(tok) <= set(";&|()\n")):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    return out


def gh_pr_commands(command: str) -> list[dict[str, Any]]:
    """Each ``gh pr create|ready`` in ``command``: {verb, args, env (leading assignments), repo, cd}.
    ``cd``: the directory a ``cd X`` earlier in the same command line moved to (the gh call runs there)."""
    out, cd = [], ""
    for words in simple_commands(command):
        env: dict[str, str] = {}
        i = 0
        while i < len(words) and _ASSIGN_RE.match(words[i]):
            k, v = words[i].split("=", 1)
            env[k] = v
            i += 1
        if i < len(words) and words[i] == "cd" and i + 1 < len(words):
            cd = words[i + 1]
            continue
        while i < len(words) and words[i] in ("rtk", "proxy", "command", "exec", "env"):
            i += 1
        if i >= len(words) or Path(words[i]).name != "gh":
            continue
        i += 1
        repo = ""
        while i < len(words) and words[i].startswith("-"):   # gh global flags before `pr`
            if words[i] in ("-R", "--repo") and i + 1 < len(words):
                repo = words[i + 1]
                i += 2
            else:
                if words[i].startswith("--repo="):
                    repo = words[i].split("=", 1)[1]
                i += 1
        if i + 1 < len(words) and words[i] == "pr" and words[i + 1] in ("create", "ready"):
            args = words[i + 2:]
            for j, a in enumerate(args):
                if a in ("-R", "--repo") and j + 1 < len(args):
                    repo = args[j + 1]
                elif a.startswith("--repo="):
                    repo = a.split("=", 1)[1]
            out.append({"verb": words[i + 1], "args": args, "env": env, "repo": repo, "cd": cd})
    return out


def _is_help(args: list[str]) -> bool:
    return any(a in ("--help", "-h") for a in args)


def is_draft(args: list[str]) -> bool:
    return any(a in ("--draft", "-d") or (a.startswith("--draft=") and a.split("=", 1)[1].lower() != "false")
               for a in args)


def undrafted_create(command: str) -> bool:
    """True when ``command`` runs ``gh pr create`` without a draft flag of its own (and not ``--help``).
    Unbalanced quotes: False (a hook never blocks on its own parse error; the shell refuses it anyway)."""
    try:
        cmds = gh_pr_commands(command)
    except ValueError:
        return False
    return any(c["verb"] == "create" and not is_draft(c["args"]) and not _is_help(c["args"]) for c in cmds)


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


WAIVER_PREFIX = "[draft gate]"


def is_answer(turn: dict[str, Any]) -> bool:
    """A session turn that can answer Orion: a reply. Not a re-send of the PR (it carries the spec/diff
    attachments again and answers nothing), and not the gate's own waiver record (``meta.waiver``; the
    body prefix covers a Hub that predates the field)."""
    meta = turn.get("meta") or {}
    return (turn.get("speaker") == "session" and not meta.get("spec") and not meta.get("waiver")
            and not str(turn.get("body") or "").startswith(WAIVER_PREFIX))


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
        if any(is_answer(t) and int(t.get("seq") or 0) > first for t in turns):
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


def send_pr_now(lib: Any, pr: dict[str, Any], *, where: str, session_id: str) -> str:
    """Send ``pr`` to Orion from the gate. The new thread id, or '' when nothing was sent."""
    try:
        message = lib.build_pr_message(pr, session_id=session_id, worktree=where, diff_timeout=6.0)
        if message is None:
            return ""
        resp = lib.send(message, timeout=5.0)
        tid = str((resp.get("thread") or {}).get("thread_id") or "")
        lib.log(f"pr_draft_gate sent_missing_pr pr={pr.get('number')} thread={tid}")
        return tid
    except Exception as exc:  # noqa: BLE001 -- Hub down again: the refusal says how to send it by hand
        lib.log(f"pr_draft_gate send_failed pr={pr.get('number')} err={type(exc).__name__}")
        return ""


def report_override(lib: Any, target: str, *, where: str, repo: str, session_id: str) -> None:
    """Tell Orion on the PR's thread that its review was waived, so the waiver is counted where Orion reads.
    Best effort: an override never fails because Hub or gh is down."""
    try:
        pr = lib.pr_info(target, cwd=where, timeout=8.0, repo=repo)
        if pr is None:
            return
        threads = list(lib.request("GET", f"/api/session-threads?pr={int(pr['number'])}", timeout=4.0).get("threads") or [])
        thread = next((t for t in threads if int(t.get("pr_number") or 0) == int(pr["number"])), None)
        if thread is None:
            return
        lib.reply(str(thread["thread_id"]), f"{WAIVER_PREFIX} Pull request #{pr['number']} was marked ready with "
                  f"{OVERRIDE}: the draft gate's check of your review was waived (Juniper's call). This is a "
                  "record, not a question.", session_id=session_id, timeout=4.0, waiver=True)
        lib.log(f"pr_draft_gate override_reported pr={pr['number']} thread={thread['thread_id']}")
    except Exception as exc:  # noqa: BLE001
        lib.log(f"pr_draft_gate override_report_failed err={type(exc).__name__}")


def check(command: str, *, cwd: str, lib: Any, now: datetime, session_id: str = "manual") -> str | None:
    """The refusal text for ``command``, or None when it may run. ``cwd``: where a bare ``gh pr ready``
    looks up the current branch's PR (a ``cd X`` earlier in the command line wins)."""
    if undrafted_create(command):
        return ("Pull requests open as drafts (CLAUDE.md, Orion review). Run the same command with --draft. "
                "Mark it ready with `gh pr ready <n>` after you answered Orion's first review.")
    try:
        cmds = gh_pr_commands(command)
    except ValueError:
        return None
    for c in cmds:
        args = c["args"]
        if c["verb"] != "ready" or "--undo" in args or _is_help(args):
            continue
        target = ready_target(args)
        where = c["cd"] if c["cd"] and Path(c["cd"]).is_absolute() else (cwd or str(lib.REPO_ROOT))
        if c["env"].get("ORION_ALLOW_PR_READY") == "1" or os.environ.get("ORION_ALLOW_PR_READY") == "1":
            lib.log(f"pr_draft_gate override args={args!r}")
            report_override(lib, target, where=where, repo=c["repo"], session_id=session_id)
            continue
        pr = lib.pr_info(target, cwd=where, timeout=8.0, repo=c["repo"])
        if pr is None:
            return (f"The draft gate could not find the pull request ({target or 'current branch'}) with gh pr view, "
                    f"so it cannot check Orion's review. Name it: gh pr ready <n>. Override: {OVERRIDE} gh pr ready <n>.")
        number = int(pr["number"])
        try:
            threads = list(lib.request("GET", f"/api/session-threads?pr={number}", timeout=4.0).get("threads") or [])
            # An older Hub ignores ?pr= and lists recent threads: take only this PR's.
            thread = next((t for t in threads if int(t.get("pr_number") or 0) == number), None)
            turns = lib.thread_view(str(thread["thread_id"]))[1] if thread else []
        except Exception as exc:  # noqa: BLE001 -- Hub down: the review cannot be checked
            return (f"Hub did not answer ({type(exc).__name__}), so the draft gate cannot check Orion's review of "
                    f"PR #{number}. Retry when Hub is up. Override: {OVERRIDE} gh pr ready {number}.")
        if thread is None:
            # The create hook never got the PR to Hub (Hub down at create, no PR URL in the output, a hook
            # error). Send it now, so Orion reviews it and the wait clock starts from this thread.
            sent = send_pr_now(lib, pr, where=where, session_id=session_id)
            return (f"PR #{number} stays a draft: Orion had no thread for it, so Orion never reviewed it. "
                    + (f"The draft gate sent it to Orion now (thread {sent}); Orion reads it at the next round-up "
                       "(every 15 min). Answer Orion's questions, then run gh pr ready again."
                       if sent else
                       f"The draft gate could not send it either. Send it by hand: python3 "
                       f"scripts/orion_session_thread.py send --pr {number} (add --note \"...\" if nothing can be read).")
                    + f" Override (Juniper's call): {OVERRIDE} gh pr ready {number}.")
        wait_h = float(os.environ.get("ORION_PR_READY_WAIT_H") or DEFAULT_WAIT_H)
        ok, why = ready_verdict(thread, turns, now=now, wait_h=wait_h)
        lib.log(f"pr_draft_gate ready pr={number} allowed={ok} why={why!r}")
        if not ok:
            return (f"PR #{number} stays a draft: {why} Answer Orion with "
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
        # The payload's cwd is where the session STARTED (often the primary checkout); the agent board knows
        # the worktree the session works in now.
        session_id = lib.session_id_from(payload)
        cwd = lib.identity(session_id).get("worktree") or str(payload.get("cwd") or "")
        reason = check(command, cwd=cwd, lib=lib, now=datetime.now(timezone.utc), session_id=session_id)
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
