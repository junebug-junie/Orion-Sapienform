"""Shared helpers for worktree hygiene tooling (prune, status, post-merge nudge,
SessionStart hook).

Deliberately one importable module rather than duplicated logic per script --
unlike scripts/git_hooks/pre-commit's shared-checkout detection, which git
forces to be byte-for-byte duplicated because git copies hook files verbatim
into .git/hooks/ with no import path back into this repo. None of the
consumers here are copied that way, so they import this normally.

Failure philosophy: git-level lookups (list_worktrees, merged_branch_set)
raise a WorktreeLibError on failure rather than silently returning an empty
result -- an empty *result* (no worktrees, no merged branches) must stay
distinguishable from a *failed lookup* (bad ref, git not on PATH, timeout),
since callers make real decisions (prune candidates, "safe to delete") off
these. Per-worktree lookups that are expected to sometimes legitimately have
no answer (disk size of a path that vanished mid-scan) still fail soft to
None -- see dir_size_bytes.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")

_GIT_TIMEOUT = 10


class WorktreeLibError(RuntimeError):
    """Raised when a git-level lookup fails outright (not: 'found nothing')."""


@dataclass
class WorktreeInfo:
    path: Path
    branch: str | None  # None for a detached-HEAD worktree
    is_main: bool


def repo_toplevel() -> str:
    """Resolves the repo root so callers (e.g. `git worktree remove`) can
    anchor with `-C` explicitly instead of relying on the ambient cwd being
    inside some worktree of the target repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorktreeLibError("'git rev-parse --show-toplevel' timed out") from exc
    if result.returncode != 0:
        raise WorktreeLibError(f"'git rev-parse --show-toplevel' failed: {result.stderr.strip()}")
    return result.stdout.strip()


def list_worktrees() -> list[WorktreeInfo]:
    """Lists every worktree registered for this repo, regardless of which
    location convention it uses (sibling directories, .worktrees/,
    .claude/worktrees/agent-<id>, etc). `git worktree list` is repo-wide,
    not scoped to the worktree it's invoked from, so this returns the same
    result no matter which worktree calls it. Git always lists the main
    worktree first.

    Raises WorktreeLibError if the underlying git command fails or times
    out -- callers should catch this and report clearly rather than let an
    empty list masquerade as "this repo has no worktrees"."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorktreeLibError(f"'git worktree list' timed out after {_GIT_TIMEOUT}s") from exc
    if result.returncode != 0:
        raise WorktreeLibError(f"'git worktree list' failed: {result.stderr.strip()}")

    infos: list[WorktreeInfo] = []
    path: Path | None = None
    branch: str | None = None
    for line in result.stdout.splitlines() + [""]:
        if line.startswith("worktree "):
            path = Path(line[len("worktree "):])
            branch = None
        elif line.startswith("branch "):
            branch = line[len("branch "):].removeprefix("refs/heads/")
        elif line == "" and path is not None:
            infos.append(WorktreeInfo(path=path, branch=branch, is_main=False))
            path = None
    if infos:
        infos[0].is_main = True
    return infos


def merged_branch_set(base: str = "origin/main") -> set[str]:
    """One `git branch --merged` call instead of one `git merge-base` per
    worktree -- at this repo's real scale (276 worktrees when this was
    written), N subprocess spawns for a single SessionStart-hook summary is
    the difference between instant and a multi-second startup stall.

    Raises WorktreeLibError on failure (bad --base ref, no such branch,
    timeout) rather than returning an empty set -- an empty set must mean
    'genuinely zero merged branches', not 'the lookup itself failed', since
    prune_merged_worktrees.py treats an empty result as "nothing to do"."""
    try:
        result = subprocess.run(
            ["git", "branch", "--merged", base, "--format=%(refname:short)"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorktreeLibError(f"'git branch --merged {base}' timed out after {_GIT_TIMEOUT}s") from exc
    if result.returncode != 0:
        raise WorktreeLibError(f"'git branch --merged {base}' failed: {result.stderr.strip()}")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def mergeable_worktrees(base: str = "origin/main") -> tuple[list[WorktreeInfo], set[str]]:
    """Shared 'which worktrees exist, which branches are merged' setup used
    by both prune_merged_worktrees.py and worktree_status.py -- factored out
    so a future change (e.g. excluding detached-HEAD worktrees) only needs
    to land in one place."""
    worktrees = [w for w in list_worktrees() if not w.is_main]
    merged_set = merged_branch_set(base=base)
    return worktrees, merged_set


def dir_size_bytes(path: Path) -> int | None:
    """Fails soft to None -- unlike list_worktrees/merged_branch_set, a
    missing size for one worktree (vanished mid-scan, permission error,
    non-GNU `du` lacking -k... though -k itself is the portable POSIX form)
    is a legitimate "don't know", not a signal callers should treat as a
    lookup-wide failure."""
    try:
        result = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return int(result.stdout.split()[0]) * 1024
    except Exception:
        return None


def all_open_prs() -> dict[str, dict] | None:
    """One `gh pr list --state open` call for ALL open PRs, mapped locally
    by branch -- not one `gh pr list --head <branch>` call per worktree.
    At this repo's real scale (276 worktrees), the per-worktree form was
    276 separate subprocess+network round trips (measured: ~2.5 minutes
    sequential) instead of 1, and a transient failure on any single one of
    those 276 calls silently reported that worktree as having no open PR
    rather than surfacing an error -- exactly the kind of false "safe to
    prune" signal this tooling must not produce.

    Returns None (not an empty dict) if the batched call itself fails, so
    callers can distinguish "there really are no open PRs" from "PR data is
    unavailable right now" and refuse to treat the latter as the former."""
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--state", "open", "--json",
             "number,state,url,headRefName", "--limit", "1000"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        prs = json.loads(result.stdout)
    except Exception:
        return None
    return {pr["headRefName"]: pr for pr in prs}


def parallel_map(fn: Callable[[_T], _R], items: list[_T], max_workers: int = 16) -> list[_R]:
    """Thin, shared wrapper so callers don't each hand-roll a ThreadPoolExecutor
    for what's structurally the same pattern: independent, I/O-bound per-item
    lookups (du, gh calls). Used for dir_size_bytes -- the only remaining
    per-item lookup that can't be batched into one call the way PR lookups
    were (see all_open_prs)."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fn, items))


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "?"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"
