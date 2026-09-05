"""GitHub PR lifecycle counts -- the ``pr_lifecycle`` piece of the codebase-mass
domain (docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md, Phase
1). Submitted/merged/closed-without-merge counts for a given time window.

**Deviation from the design spec's original assumption, confirmed live
2026-07-30:** the spec proposed reading ``items_total`` from "the pre-trim
``github_compactor`` fetch payload" to bypass ``trim_github_compactor_input()``'s
``MAX_DIGEST_INPUT_PRS`` digest-input cap. That fetch
(``services/orion-cortex-exec/app/verb_adapters.py::GithubRecentPullRequestsVerb``)
has two narrower limits of its own, upstream of the digest item list: it calls
the GitHub REST API with a single ``per_page`` request (never paginated further;
raised to 100 alongside the digest budget so a ~30-merge day can reach the
summarizer), and it unconditionally drops every PR with no ``merged_at``
(``if not merged_at: continue``) -- so it structurally can never report
"submitted" (opened) or "closed-without-merge" counts, only merged ones,
regardless of window size. Reusing it would still miss open/closed-without-merge
counts. This module fetches independently via the ``gh`` CLI instead
(already authenticated in this repo's own dev/CI environments, same command
this repo's own PR-management workflow already uses) -- not a reuse of
``github_compactor``'s fetch, a deliberate divergence from it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PrLifecycleDelta:
    since: datetime
    until: datetime
    submitted_count: int
    merged_count: int
    closed_without_merge_count: int
    submitted_numbers: tuple[int, ...] = field(default_factory=tuple)
    merged_numbers: tuple[int, ...] = field(default_factory=tuple)
    closed_without_merge_numbers: tuple[int, ...] = field(default_factory=tuple)
    # True when the fetch appears to have hit its `limit` (see
    # pr_lifecycle_delta()'s `fetch_limit` parameter) AND the oldest fetched
    # `updated_at` is not older than `since` -- a sign the fetch may not reach
    # far enough back to see every real submitted/merged/closed event in the
    # window, so counts here could under-report. See pr_lifecycle_delta()'s
    # docstring for why `updated_at`, not `created_at`, is the right basis.
    possibly_truncated: bool = False


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def fetch_recent_prs(owner: str, repo: str, *, limit: int = 200) -> list[dict]:
    """Shells out to ``gh pr list --state all``, normalized to
    ``number``/``created_at``/``merged_at``/``closed_at``/``updated_at``/``state``
    (parsed ``datetime`` values, not raw strings). Pure I/O, no windowing logic --
    ``pr_lifecycle_delta()`` below does the actual categorization, mirroring
    ``git_delta.py``'s split between fetch and pure computation.

    **Sorted by ``updated_at`` descending (``--search "sort:updated-desc"``), not
    plain ``gh pr list``'s default ``created_at`` descending -- deliberate,
    confirmed live 2026-07-30.** Plain ``gh pr list`` (no ``--search``/``--sort``
    flag exists per its own ``--help``) always sorts newest-*created*-first,
    which silently misses old-but-recently-active PRs: a PR opened long ago and
    merged today ranks near the bottom of a created-desc list and can fall
    outside ``limit`` even though its ``merged_at`` is brand new. Every one of
    this domain's three tracked events (submit/merge/close) bumps GitHub's own
    ``updated_at`` on the PR, so sorting by that instead means the fetch's
    completeness (or lack of it) can be judged against a single timestamp that
    actually bounds all three buckets, not just the submitted one. ``--search``
    routes through GitHub's Search API rather than the plain list endpoint --
    real, but its index can lag actual state by up to a few minutes; not a
    concern at this domain's real cadence (interval-driven ticks, not
    sub-minute polling), worth knowing if this is ever adapted for one.

    ``limit`` must be generous enough to reach back past the oldest event the
    caller cares about, or real events will be silently missed -- the same
    caller-responsibility shape ``git_churn_delta``'s ``prev_sha``/``head_sha``
    range has, just expressed as a row count instead of a commit range because
    the GitHub CLI has no equivalent of ``git rev-list``'s exact-range query
    for PRs. Pass this same ``limit`` to ``pr_lifecycle_delta()`` so it can
    tell "reached the real end of PR history" apart from "hit the fetch cap."
    """
    result = subprocess.run(
        [
            "gh", "pr", "list",
            "--repo", f"{owner}/{repo}",
            "--state", "all",
            "--search", "sort:updated-desc",
            "--json", "number,createdAt,mergedAt,closedAt,updatedAt,state",
            "--limit", str(limit),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh pr list failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    raw = json.loads(result.stdout or "[]")
    if not isinstance(raw, list):
        raise RuntimeError(f"gh pr list returned a non-list JSON body: {raw!r}")
    return [
        {
            "number": item.get("number"),
            "created_at": _parse_ts(item.get("createdAt")),
            "merged_at": _parse_ts(item.get("mergedAt")),
            "closed_at": _parse_ts(item.get("closedAt")),
            "updated_at": _parse_ts(item.get("updatedAt")),
            "state": item.get("state"),
        }
        for item in raw
        if isinstance(item, dict)
    ]


def pr_lifecycle_delta(
    prs: list[dict],
    since: datetime,
    until: datetime,
    *,
    fetch_limit: int | None = None,
) -> PrLifecycleDelta:
    """Categorize already-fetched PR records (``fetch_recent_prs()``'s shape)
    into submitted/merged/closed-without-merge counts within the half-open
    window ``[since, until)``.

    ``fetch_limit`` should be the same ``limit`` passed to
    ``fetch_recent_prs()`` -- it lets ``possibly_truncated`` distinguish "the
    fetch returned everything GitHub has" (``len(prs) < fetch_limit``, nothing
    missing regardless of how far back events go) from "the fetch may have
    been cut off before reaching ``since``" (``len(prs) >= fetch_limit`` *and*
    the oldest fetched ``updated_at`` doesn't reach back past ``since``).
    Without this, a window wider than any real PR history would falsely flag
    truncation forever (every fetch's oldest event would be "recent" relative
    to an ancient ``since``, even though nothing was actually cut off).

    **"Special time horizon handling" (design spec idea #4), operationalized as
    half-open interval tiling, not a per-event dedup store.** A PR event
    (created/merged/closed) is counted exactly once as long as the caller's
    successive windows tile contiguously (``until`` of one tick becomes
    ``since`` of the next, no gap, no overlap) -- the interval boundary itself
    is the dedup mechanism, the same role ``git_churn_delta``'s ``prev_sha`` ->
    ``head_sha`` chaining plays for commits. No PR-number-keyed state needs to
    be persisted between ticks for this alone; a PR merged near a tick boundary
    is counted in whichever window its ``merged_at`` timestamp actually falls
    in, never both, never neither -- unlike a hard daily cutoff (idea #4's
    named alternative), which can double-count or miss events that straddle a
    calendar-day boundary depending on exactly when a poll happens to run.

    A single PR can appear in more than one bucket for the *same* window (e.g.
    opened and merged within one short window) -- that is real, not a bug: the
    three buckets measure independent lifecycle transitions, not a partition of
    PRs into mutually exclusive states.

    ``closed_without_merge`` requires both ``closed_at`` in-window and
    ``merged_at is None`` -- a merged PR's ``closed_at`` is always set too
    (GitHub sets both timestamps on merge), so without the ``merged_at is
    None`` guard every merge would double-count as a close-without-merge.
    """
    submitted: list[int] = []
    merged: list[int] = []
    closed_without_merge: list[int] = []
    oldest_updated_at: datetime | None = None

    for pr in prs:
        number = pr.get("number")
        created_at = pr.get("created_at")
        merged_at = pr.get("merged_at")
        closed_at = pr.get("closed_at")
        updated_at = pr.get("updated_at")

        if updated_at is not None and (oldest_updated_at is None or updated_at < oldest_updated_at):
            oldest_updated_at = updated_at

        if created_at is not None and since <= created_at < until:
            submitted.append(number)
        if merged_at is not None and since <= merged_at < until:
            merged.append(number)
        if closed_at is not None and merged_at is None and since <= closed_at < until:
            closed_without_merge.append(number)

    fetch_appears_capped = fetch_limit is not None and len(prs) >= fetch_limit
    possibly_truncated = (
        fetch_appears_capped
        and oldest_updated_at is not None
        and oldest_updated_at >= since
    )

    return PrLifecycleDelta(
        since=since,
        until=until,
        submitted_count=len(submitted),
        merged_count=len(merged),
        closed_without_merge_count=len(closed_without_merge),
        submitted_numbers=tuple(submitted),
        merged_numbers=tuple(merged),
        closed_without_merge_numbers=tuple(closed_without_merge),
        possibly_truncated=possibly_truncated,
    )
