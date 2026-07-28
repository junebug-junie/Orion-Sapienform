#!/usr/bin/env python3
"""orion-edit-guard: PreToolUse hook that blocks Edit/Write/NotebookEdit calls
from writing directly into the shared/primary checkout of this repo.

Companion to scripts/hooks/destructive_git_guard.py, which blocks destructive
*commands* (reset --hard, clean -f, checkout/switch --force) run via Bash
against the shared checkout. That hook has no coverage at all for the much
more common failure mode: an agent simply calling Edit/Write on a path under
the shared checkout instead of a worktree. Nothing previously stopped this --
the pre-commit hook only fires at commit time, by which point the working
tree is already dirty and can block a teammate's `git pull` (confirmed live,
2026-07-28: a prior session's uncommitted Edit-tool changes to this exact
checkout sat dirty for the rest of that session, then blocked `git pull` on
an unrelated incoming merge until manually stashed and reconciled).

Reuses _is_shared_checkout's exact detection from destructive_git_guard.py
(primary checkout's --git-dir IS its --git-common-dir; a linked worktree's
differs) rather than reimplementing it.

Escape hatch: ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 in the hook process's
environment -- same variable destructive_git_guard.py uses, so one export
covers both guards for a genuinely intentional shared-checkout session
(e.g. editing repo-root tooling that must be tested from the primary
checkout, or scripted git-hook installers themselves).

Gitignored files are exempt: they cannot dirty tracked state or block a
teammate's `git pull`, which is the actual harm this hook exists to
prevent -- blocking edits to e.g. .claude/settings.local.json or scratch
files would be a pure false positive with no corresponding protection.

Known gaps (disclosed, not silent), inherited from the same tradeoffs
destructive_git_guard.py already documents:
  - Not scoped to this repo specifically -- any primary (non-worktree) git
    checkout of any repo triggers this, same as the sibling hook.
  - _is_shared_checkout fails open (allows) on any git-subprocess error,
    non-zero exit, or its 5s timeout.
  - Only fires for Claude Code sessions that load this repo's
    .claude/settings.json -- a different tool or a human editing directly
    is unprotected.
  - Symlinks are not resolved (normpath, not realpath) -- matches the
    logical `cd ... && pwd` semantics the shell-based git hooks use.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def _is_shared_checkout(target_dir: str) -> bool:
    """Mirrors destructive_git_guard.py's _is_shared_checkout exactly."""
    try:
        proc = subprocess.run(
            ["git", "-C", target_dir, "rev-parse", "--git-dir", "--git-common-dir"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    lines = proc.stdout.splitlines()
    if len(lines) != 2:
        return False
    git_dir_raw, common_dir_raw = lines
    gd = os.path.normpath(os.path.join(target_dir, git_dir_raw.strip()))
    cd = os.path.normpath(os.path.join(target_dir, common_dir_raw.strip()))
    return gd == cd


def _is_gitignored(target_dir: str, file_path: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", target_dir, "check-ignore", "-q", file_path],
            capture_output=True, timeout=5,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _escape_hatch_set() -> bool:
    return os.environ.get("ORION_ALLOW_SHARED_CHECKOUT_WRITE") == "1"


def main() -> None:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return
    file_path = str(tool_input.get("file_path") or "")
    if not file_path.strip():
        return

    file_path = os.path.normpath(os.path.expanduser(file_path))
    target_dir = os.path.dirname(file_path) or os.getcwd()

    if not _is_shared_checkout(target_dir):
        return
    if _is_gitignored(target_dir, file_path):
        return
    if _escape_hatch_set():
        return

    decision = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"[edit-guard] Blocked: writing to {file_path} in the shared/primary "
                f"checkout. This repo's policy is worktrees-only for writing/building/"
                f"running code -- concurrent sessions have collided over shared-checkout "
                f"edits before. Redo this in a worktree instead: "
                f"git worktree add ../Orion-Sapienform-<name> -b <type>/<name>. "
                f"If this is genuinely intentional, set "
                f"ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 in the environment."
            ),
        }
    }
    sys.stdout.write(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
