"""Pure git-diff-hunk extraction and commit-prefix classification for the
doc-semantic-drift domain (docs/superpowers/specs/2026-07-30-doc-semantic-
drift-design.md). Shells out to ``git``, holds no state of its own -- same
split as ``git_delta.py``/``pr_lifecycle.py``/``graph_delta.py`` in this
module: the caller (``orion-cocreation-signals``'s producer) owns polling
and publishing.

``diff_hunks()`` is the calibrated fix from
``scripts/analysis/measure_doc_semantic_drift.py`` (real replay evidence:
docs/superpowers/pr-reports/2026-08-11-doc-semantic-drift-diff-scoped-
embedding.md) -- embedding only the changed hunk lines, not the whole
document, is what makes the embedding-diff signal actually separate trivial
edits from real ones. Moved here, unchanged in behavior, so the live
producer and the calibration script share one implementation instead of
two copies drifting apart.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

_CONVENTIONAL_PREFIXES = (
    "feat", "fix", "chore", "docs", "test", "refactor", "perf", "build", "ci",
)


def conventional_commit_prefix(subject: str) -> str | None:
    """The free tier: a self-declared conventional-commit type prefix
    (``docs(x): ...``, ``fix: ...``), or ``None`` if the subject doesn't use
    one. Weak feature, per the design doc's own "Missing questions" -- a
    ``docs:`` commit is self-declared, not verified against what actually
    changed."""
    head = subject.split(":", 1)[0].strip()
    kind = head.split("(", 1)[0].strip().lower()
    return kind if kind in _CONVENTIONAL_PREFIXES else None


# git's --name-status status letters -> this domain's own change_kind
# vocabulary. C (copied) and R (renamed) both present as a full add of the
# new path (no -M/-C passed to `git diff`), so they read as "added" here --
# matching what the hunk text actually looks like, which is what gets
# scored, rather than what git technically calls the operation.
_STATUS_TO_CHANGE_KIND = {"A": "added", "C": "added", "R": "added", "M": "modified", "D": "deleted"}

# Generated *.md that is not authored prose and must not be scored as doc
# drift. `graphify-out/` is machine-regenerated on a post-commit hook, so
# its "semantic drift" measures a tool's output churn, not co-creation --
# and GRAPH_REPORT.md alone produces ~182 removed / ~175 added 1200-char
# chunks (measured over 400 real commits, 2026-08-12). At the embedding
# host's real ~0.6s serial throughput that is minutes of work per commit
# against a 30s per-request timeout, so the file could never actually be
# scored while still starving the shared embedding host that live chat
# memory depends on.
_EXCLUDED_DOC_PATHSPECS = (":(exclude)graphify-out/**",)


def changed_doc_files_with_status(
    prev_sha: str, head_sha: str, repo_path: str | Path = "."
) -> list[tuple[str, str]]:
    """(change_kind, path) for every real ``*.md`` file touched between two
    commits -- added/copied/modified/renamed, not deleted (a deleted doc has
    no "after" state to score drift against).

    Reads git's own ``--name-status`` letter rather than inferring the kind
    from whether a hunk side came back empty: an added file and a modified
    file whose entire content was replaced can produce similar-looking
    hunks, but they mean different things for a drift score (see
    ``DocSemanticDriftV1.change_kind``).

    A malformed/unparseable status line is skipped rather than guessed at --
    ``git diff --name-status`` emits ``<STATUS>\\t<path>``, and a line
    without that shape isn't a real file entry."""
    result = subprocess.run(
        [
            "git", "diff", "--name-status", "--diff-filter=ACMR", prev_sha, head_sha,
            "--", "*.md", *_EXCLUDED_DOC_PATHSPECS,
        ],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    entries: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        # R/C statuses carry a similarity score (e.g. "R100") and a second
        # path column (old -> new); the *last* column is the current path.
        status_letter = parts[0].strip()[:1].upper()
        path = parts[-1].strip()
        change_kind = _STATUS_TO_CHANGE_KIND.get(status_letter)
        if change_kind is None or not path:
            continue
        entries.append((change_kind, path))
    return entries




def diff_hunks(
    before_rev: str, after_rev: str, path: str, repo_path: str | Path = "."
) -> tuple[str, str]:
    """(removed_text, added_text) -- just the changed lines between
    ``before_rev`` and ``after_rev`` for one file, not the whole
    before/after document. See module docstring for why this, not
    whole-document embedding. ``before_rev``/``after_rev`` are any two real
    refs (a single commit's own parent for a per-commit diff, or a wider
    ``prev_sha``/``head_sha`` poll window for a range that covers several
    commits -- see ``doc_semantic_drift_changes``).

    File-header lines (``--- a/...``, ``+++ b/...``) and hunk markers
    (``@@ ... @@``) are excluded -- only real content lines. A pure addition
    yields an empty ``removed_text``; a pure deletion yields an empty
    ``added_text`` -- both real, meaningful states, not errors.

    Known, currently-unhit limitation (unchanged from the calibration
    script this was moved from): a renamed file (no ``-M`` passed to
    ``git diff``) shows as a full delete-at-old-path + full-add-at-new-path,
    degenerating hunk text back to whole-document size. A binary-file diff
    produces no ``+``/``-`` lines at all, silently yielding two empty
    strings. Neither is guarded against here -- callers filtering to
    ``*.md`` files make both cases unlikely but not impossible (a binary
    asset with a ``.md`` extension, a renamed doc)."""
    result = subprocess.run(
        ["git", "diff", "--no-color", "--unified=0", before_rev, after_rev, "--", path],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    removed: list[str] = []
    added: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return "\n".join(removed), "\n".join(added)


@dataclass(frozen=True)
class DocHunkChange:
    """One real doc file's real diff hunk for one real commit -- what the
    producer needs to request two embeddings and publish one
    ``DocSemanticDriftV1`` event."""

    sha: str
    path: str
    commit_subject: str
    commit_prefix: str | None
    hunk_removed: str
    hunk_added: str
    # Real git status for this file in the scored range -- see
    # `changed_doc_files_with_status()` and `DocSemanticDriftV1.change_kind`.
    change_kind: str | None = None


def _commit_subject(sha: str, repo_path: str | Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%s", sha],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def doc_semantic_drift_changes(
    prev_sha: str, head_sha: str, repo_path: str | Path = "."
) -> list[DocHunkChange]:
    """One ``DocHunkChange`` per real ``*.md`` file touched at ``head_sha``
    that also changed somewhere in ``(prev_sha, head_sha]`` -- real commit
    range, real files, real hunks. Attributes each file's hunk to
    ``head_sha`` (the tick's real, current commit) rather than trying to
    walk every intermediate commit in the range separately -- a real, small
    simplification: a doc touched by multiple commits within one poll
    window gets one hunk-diff request covering its net change since the
    last tick, not one per intermediate commit. Matches
    ``git_delta_loop``'s own "diffs the whole range, not commit-by-commit"
    convention for the same reason (self-healing across a missed poll)."""
    entries = changed_doc_files_with_status(prev_sha, head_sha, repo_path)
    subject = _commit_subject(head_sha, repo_path)
    prefix = conventional_commit_prefix(subject)
    changes = []
    for change_kind, path in entries:
        removed, added = diff_hunks(prev_sha, head_sha, path, repo_path)
        # Skip a file whose git-diff produced no real +/- lines at all
        # (e.g. a rename or binary-asset edge case the module docstring
        # names) -- nothing real to score.
        if not removed and not added:
            continue
        changes.append(
            DocHunkChange(
                sha=head_sha,
                path=path,
                commit_subject=subject,
                commit_prefix=prefix,
                hunk_removed=removed,
                hunk_added=added,
                change_kind=change_kind,
            )
        )
    return changes
