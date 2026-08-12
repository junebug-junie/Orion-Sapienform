"""Schema for the self-study semantic enrichment request event.

Producer: `scripts/git_hooks/post-commit` (a lightweight one-shot publish
from a host/shell context after a qualifying commit -- see
`scripts/self_study_enrichment_hook.py`).

Consumer: `services/orion-self-study-enrichment` -- spawns a real `claude -p`
subprocess to synthesize a grounded "what is this and why" summary for the
affected cluster(s), evidence-in/prose-out, and caches the result to disk.

This event carries enough to identify *what structurally changed* (real
git-diff-derived facts from `orion/structural_mass/git_delta.py`, plus the
list of touched self-study-relevant paths) -- it does not carry the prose
itself; that is produced downstream by the consumer, not by the hook.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SelfStudyEnrichmentRequestV1(BaseModel):
    """Requests a semantic enrichment run for the self-study cluster(s)
    touched by a real commit. One event per qualifying commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: str = Field(..., description="Absolute path to the repo checkout that produced this commit.")
    prev_sha: str = Field(..., description="SHA the delta is computed from (last enrichment run, or HEAD~1 on cold start).")
    head_sha: str = Field(..., description="SHA of the commit that triggered this request.")
    commit_count: int = Field(..., ge=0)
    files_changed: int = Field(..., ge=0)
    lines_changed: int = Field(..., ge=0)
    touched_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Repo-relative paths (from `git diff --name-only prev_sha head_sha`) that matched the self-study-relevant path patterns and triggered this request.",
    )
    requested_at: str = Field(..., description="ISO-8601 UTC timestamp the hook published this event.")
