#!/usr/bin/env python3
"""orion-git-safety-guard: PreToolUse hook that blocks destructive git
commands (git clean -f*, git reset --hard, git checkout/switch --force)
from running against the shared/primary checkout of this repo.

Companion to scripts/git_hooks/pre-commit, which only gates `git commit`.
That hook cannot help here: these commands destroy uncommitted files
*before* any commit happens, so a commit-time gate is too late -- the
damage is already done. This hook gates at tool-call time instead, using
Claude Code's PreToolUse block contract (see AGENTS.md section 2, "Clean
git and worktree rules").

Deliberately scoped to Claude Code, not vendor-neutral: unlike `git
commit`, git has no hook point at all for `clean`/`reset`/`checkout`, so
there is no equivalent of scripts/git_hooks/pre-commit for these commands
-- the only way to get the same tool-independent guarantee is to
intercept the `git` binary itself (e.g. a PATH-shadowing wrapper), which
is a materially bigger change than this file. That wrapper is deferred,
not abandoned -- it's next up once the underlying question of *why*
agents end up in the shared checkout at all (worktree-discipline
noncompliance) is addressed; this hook is a real, live block in the
meantime, not just a placeholder.

Escape hatch: ORION_ALLOW_SHARED_CHECKOUT_WRITE=1, same as the commit
hook -- either set in the hook process's own environment, or as a leading
env-var assignment on the SAME statement as the destructive command
(e.g. `ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git reset --hard`), matching
how a real shell scopes an inline assignment to the one command it
prefixes.

Known gaps (disclosed, not silent):
  - Shell comments (`# ...`) are not stripped. A `&&`/`;`/`|` appearing
    after a `#` on the same line can cause an incorrect statement split.
    This errs toward over-blocking (a harmless command misread as two
    statements), not under-blocking.
  - Quote-stripping (used so quoted text like a commit message can't be
    mistaken for a real git invocation or an inline escape-hatch
    assignment) is a balanced-quote heuristic, not a real shell
    tokenizer -- it does not handle escaped quotes or nesting.
  - Variable expansion ($VAR, $(...), backticks) is not performed. A `cd
    "$VAR"` or `git -C "$VAR"` resolves to the literal string, which
    almost always fails the "is this a real directory" check and so
    fails OPEN (allowed), not closed.
  - `_is_shared_checkout` fails open (allows) on any git-subprocess
    error, non-zero exit, or its 5s timeout.
  - `git push --force`/`--force-with-lease` and `git branch -D` are
    intentionally out of scope -- push affects the remote regardless of
    which checkout runs it (not a "shared checkout" problem specifically)
    and git itself already refuses to delete a branch checked out in
    another worktree.
  - `git switch -C` / `git checkout -B` (force-create-and-reset branch
    variants) are not detected; only the `-f`/`--force` forms of
    checkout/switch are.
  - This hook only fires for Claude Code sessions that load this repo's
    .claude/settings.json. A different tool, a plain terminal, or a human
    typing the command directly is unprotected -- see the "Deliberately
    scoped" note above.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

_STATEMENT_SPLIT = re.compile(r"&&|;|\|")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_FORCE_FLAG_RE = re.compile(r"(?:^|\s)-[a-zA-Z]*f[a-zA-Z]*(?:\s|$)")
_DRYRUN_FLAG_RE = re.compile(r"(?:^|\s)-[a-zA-Z]*n[a-zA-Z]*(?:\s|$)")
_CD_STATEMENT = re.compile(r'^cd\s+(["\']?)([^"\']+)\1\s*$')
_GIT_DASH_C = re.compile(r'\bgit\s+-C\s+(["\']?)([^"\'\s]+)\1')
_GIT_CLEAN = re.compile(r"\bgit\b(?:\s+-C\s+\S+)?\s+clean\b")
_GIT_RESET_HARD = re.compile(r"\bgit\b(?:\s+-C\s+\S+)?\s+reset\b.*--hard\b", re.DOTALL)
_GIT_CHECKOUT = re.compile(r"\bgit\b(?:\s+-C\s+\S+)?\s+checkout\b")
_GIT_SWITCH = re.compile(r"\bgit\b(?:\s+-C\s+\S+)?\s+switch\b")
# Matches ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 only as a leading env-var
# assignment on the statement (optionally after other VAR=val assignments),
# mirroring real shell scoping of an inline prefix assignment.
_ESCAPE_HATCH_PREFIX = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*ORION_ALLOW_SHARED_CHECKOUT_WRITE=1(?:\s|$)"
)


def _strip_quoted(s: str) -> str:
    """Blank out single/double-quoted spans (same length, so positions are
    preserved) so text inside a commit message, echoed string, etc. can't
    be mistaken for a real git invocation, a directory override, or an
    inline escape-hatch assignment. A best-effort heuristic, not a shell
    tokenizer -- see the module docstring's Known gaps."""
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), s)


def _split_statements(command: str) -> list[tuple[str, str]]:
    """Splits on &&/;/| using a quote-stripped view (so a quoted string
    containing one of those characters, e.g. a commit message, doesn't
    create a bogus statement boundary), then returns (original, stripped)
    pairs aligned by identical character offsets -- the stripped view is
    used for pattern matching, the original for extracting real path
    values out of quoted arguments."""
    masked = _strip_quoted(command)
    spans: list[tuple[str, str]] = []
    pos = 0
    for m in _STATEMENT_SPLIT.finditer(masked):
        spans.append((command[pos:m.start()], masked[pos:m.start()]))
        pos = m.end()
    spans.append((command[pos:], masked[pos:]))
    return [(o.strip(), s.strip()) for o, s in spans if s.strip()]


def _statement_is_destructive(statement: str) -> str | None:
    """`statement` must already be quote-stripped."""
    if _GIT_CLEAN.search(statement):
        is_dry_run = _DRYRUN_FLAG_RE.search(statement) or "--dry-run" in statement
        is_forced = _FORCE_FLAG_RE.search(statement) or "--force" in statement
        # git clean treats -n as authoritative even combined with -f:
        # nothing is deleted, so a dry-run preview is not destructive.
        if is_forced and not is_dry_run:
            return "git clean (force flag, no dry-run)"
    if _GIT_RESET_HARD.search(statement):
        return "git reset --hard"
    if _GIT_CHECKOUT.search(statement) or _GIT_SWITCH.search(statement):
        is_forced = _FORCE_FLAG_RE.search(statement) or "--force" in statement
        if is_forced:
            kind = "checkout" if _GIT_CHECKOUT.search(statement) else "switch"
            return f"git {kind} (force flag)"
    return None


def _resolve(base: str, path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return os.path.normpath(path)


def _is_shared_checkout(target_dir: str) -> bool:
    """Mirrors the detection in scripts/git_hooks/pre-commit and
    scripts/safe_docker_build.sh: the primary checkout's --git-dir IS the
    --git-common-dir; a linked worktree's differs. Uses normpath (not
    realpath) so symlinks are treated the same way the shell hooks' logical
    `cd ... && pwd` treats them -- i.e. not resolved."""
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


def _escape_hatch_set(statement: str) -> bool:
    if os.environ.get("ORION_ALLOW_SHARED_CHECKOUT_WRITE") == "1":
        return True
    return bool(_ESCAPE_HATCH_PREFIX.match(statement))


def _evaluate(cwd: str, command: str) -> tuple[str, str] | None:
    """Walks statements left-to-right, tracking a running current directory
    across `cd <path>` statements (so `cd a && cd b && git reset --hard`
    resolves correctly, not just a single leading cd). Returns
    (reason, target_dir) for the first destructive statement that targets
    the shared checkout and isn't escape-hatched, else None."""
    current_dir = cwd
    for original, statement in _split_statements(command):
        cd_match = _CD_STATEMENT.match(original)
        if cd_match:
            current_dir = _resolve(current_dir, cd_match.group(2).strip())
            continue

        reason = _statement_is_destructive(statement)
        if reason is None:
            continue

        dash_c = _GIT_DASH_C.search(original)
        target_dir = (
            _resolve(current_dir, dash_c.group(2).strip()) if dash_c else current_dir
        )

        if not _is_shared_checkout(target_dir):
            continue
        if _escape_hatch_set(statement):
            continue
        return reason, target_dir
    return None


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
    command = str(tool_input.get("command") or "")
    if not command.strip():
        return

    cwd = str(payload.get("cwd") or os.getcwd())
    result = _evaluate(cwd, command)
    if result is None:
        return
    reason, target_dir = result

    decision = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"[git-safety] Blocked: {reason} in the shared/primary checkout "
                f"({target_dir}). This repo's policy is worktrees-only for "
                f"writing/building/running code -- concurrent sessions have "
                f"silently destroyed each other's uncommitted work this way "
                f"before. Redo this in a worktree instead: "
                f"git worktree add ../Orion-Sapienform-<name> -b <type>/<name>. "
                f"If this is genuinely intentional, prefix the command with "
                f"ORION_ALLOW_SHARED_CHECKOUT_WRITE=1."
            ),
        }
    }
    sys.stdout.write(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
