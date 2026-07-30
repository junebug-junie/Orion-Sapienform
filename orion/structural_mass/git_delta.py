"""Pure git-diff summary between two commits -- the ``git_churn_delta`` piece of
the codebase-mass domain (docs/superpowers/specs/2026-07-30-codebase-mass-signal-
design.md, Phase 1). Shells out to ``git``, holds no state of its own: the caller
(the not-yet-built ``orion-cocreation-signals`` producer) owns persisting
``head_sha`` as the next call's ``prev_sha``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitChurnDelta:
    prev_sha: str
    head_sha: str
    commit_count: int
    files_added: int
    files_deleted: int
    files_modified: int
    lines_added: int
    lines_removed: int

    @property
    def files_changed(self) -> int:
        return self.files_added + self.files_deleted + self.files_modified

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_removed


def _run_git(args: list[str], repo_path: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # subprocess.CalledProcessError's own __str__ drops stderr (just "returned
        # non-zero exit status N") -- surface git's real message (e.g. "fatal:
        # Invalid revision range") in the exception text itself, since a caller
        # that logs str(exc) (the common case) would otherwise lose it.
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def git_churn_delta(prev_sha: str, head_sha: str, repo_path: str | Path = ".") -> GitChurnDelta:
    """Commit count, file add/delete/modify counts, and net line churn between
    ``prev_sha`` and ``head_sha`` (``git diff prev_sha head_sha``, i.e. the total
    working-tree difference, not a squash of individual per-commit diffs -- matches
    how every other domain in this module diffs two snapshots, not a sum of
    intermediate states).

    ``prev_sha == head_sha`` returns an explicit all-zero delta rather than
    shelling out -- a real no-op tick (nothing changed since the last check), not
    a degenerate/uncomputed value. Per this repo's "no empty-shell cognition"
    rule, a real, deliberately-computed zero is not the same failure as a value
    that was never actually measured.

    Trigger shape is caller-driven and tick-interval-based, not git-hook-driven
    (see the design spec's "self-healing by construction" note): a commit made on
    another node, via Cursor, or without this repo's ``post-commit`` hook firing
    doesn't lose anything -- the next successful check just diffs a larger range.
    """
    repo = Path(repo_path)
    if prev_sha == head_sha:
        return GitChurnDelta(
            prev_sha=prev_sha,
            head_sha=head_sha,
            commit_count=0,
            files_added=0,
            files_deleted=0,
            files_modified=0,
            lines_added=0,
            lines_removed=0,
        )

    commit_count = int(
        _run_git(["rev-list", "--count", f"{prev_sha}..{head_sha}"], repo).strip()
    )

    # --find-renames: plain `git diff` does NOT detect renames/copies by default
    # (confirmed live, git 2.43.0 -- a pure rename with no flags reports as an
    # unrelated A+D pair, not a single R line). Without this flag, every rename
    # would double-count as one added file and one deleted file instead of one
    # modified file -- exactly the "net new/removed file" miscount this bucketing
    # exists to avoid. `--find-copies` is deliberately not added: copy detection
    # is more expensive (it has to scan unchanged blobs too, not just the diff)
    # and this domain has no real need to distinguish "copied" from "added" for a
    # mass-sensing signal, unlike a rename, which is not new mass.
    files_added = files_deleted = files_modified = 0
    for line in _run_git(
        ["diff", "--find-renames", "--name-status", prev_sha, head_sha], repo
    ).splitlines():
        if not line.strip():
            continue
        status = line.split("\t", 1)[0]
        if status.startswith("A"):
            files_added += 1
        elif status.startswith("D"):
            files_deleted += 1
        else:
            # M (modified), R (renamed, now detected via --find-renames above),
            # T (type-changed) all count as "modified" for mass-sensing purposes
            # -- a rename isn't a net new or removed file, and the spec names
            # only three buckets (added/deleted/modified).
            files_modified += 1

    lines_added = lines_removed = 0
    for line in _run_git(
        ["diff", "--find-renames", "--numstat", prev_sha, head_sha], repo
    ).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        added, removed = parts[0], parts[1]
        # Binary files report "-" for both columns -- not line-countable, skip.
        if added != "-":
            lines_added += int(added)
        if removed != "-":
            lines_removed += int(removed)

    return GitChurnDelta(
        prev_sha=prev_sha,
        head_sha=head_sha,
        commit_count=commit_count,
        files_added=files_added,
        files_deleted=files_deleted,
        files_modified=files_modified,
        lines_added=lines_added,
        lines_removed=lines_removed,
    )
