"""Unit tests for ``orion/structural_mass/pr_lifecycle.py``.

``pr_lifecycle_delta`` is tested directly against synthetic PR records (no
network) -- the same fetch/compute split ``git_delta.py`` uses, so the pure
categorization logic doesn't need a real ``gh`` call to verify. ``fetch_recent_prs``
is tested against a stubbed ``subprocess.run`` for its JSON-normalization and
error-handling behavior only, not real GitHub I/O (that's what
``scripts/analysis/measure_pr_lifecycle.py`` is for, per this program's
measure-before-minting discipline -- run against real data, not in the unit
suite)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pytest

from orion.structural_mass.pr_lifecycle import (
    fetch_recent_prs,
    pr_lifecycle_delta,
)


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def _pr(
    number: int,
    *,
    created_at: str | None = None,
    merged_at: str | None = None,
    closed_at: str | None = None,
    updated_at: str | None = None,
    state: str = "OPEN",
) -> dict:
    # updated_at defaults to the latest of closed/merged/created -- matches
    # GitHub's real behavior (every one of those events bumps updated_at).
    resolved_updated_at = updated_at or closed_at or merged_at or created_at
    return {
        "number": number,
        "created_at": _dt(created_at) if created_at else None,
        "merged_at": _dt(merged_at) if merged_at else None,
        "closed_at": _dt(closed_at) if closed_at else None,
        "updated_at": _dt(resolved_updated_at) if resolved_updated_at else None,
        "state": state,
    }


# -- pr_lifecycle_delta --------------------------------------------------


def test_empty_prs_is_all_zero() -> None:
    result = pr_lifecycle_delta([], since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.submitted_count == 0
    assert result.merged_count == 0
    assert result.closed_without_merge_count == 0
    assert result.possibly_truncated is False


def test_counts_submitted_pr_in_window() -> None:
    prs = [_pr(1, created_at="2026-07-01T12:00:00", state="OPEN")]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.submitted_count == 1
    assert result.submitted_numbers == (1,)
    assert result.merged_count == 0
    assert result.closed_without_merge_count == 0


def test_counts_merged_pr_in_window() -> None:
    prs = [
        _pr(
            2,
            created_at="2026-06-30T10:00:00",  # before window -- not "submitted" this tick
            merged_at="2026-07-01T12:00:00",
            closed_at="2026-07-01T12:00:00",
            state="MERGED",
        )
    ]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.submitted_count == 0
    assert result.merged_count == 1
    assert result.merged_numbers == (2,)
    assert result.closed_without_merge_count == 0


def test_merged_pr_does_not_also_count_as_closed_without_merge() -> None:
    """GitHub sets closed_at on every merge too -- without the merged_at-is-None
    guard, every merge would double-count as a close-without-merge."""
    prs = [
        _pr(
            3,
            created_at="2026-07-01T09:00:00",
            merged_at="2026-07-01T12:00:00",
            closed_at="2026-07-01T12:00:00",
            state="MERGED",
        )
    ]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.merged_count == 1
    assert result.closed_without_merge_count == 0


def test_counts_closed_without_merge() -> None:
    prs = [
        _pr(
            4,
            created_at="2026-06-30T09:00:00",
            merged_at=None,
            closed_at="2026-07-01T12:00:00",
            state="CLOSED",
        )
    ]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.closed_without_merge_count == 1
    assert result.closed_without_merge_numbers == (4,)
    assert result.merged_count == 0


def test_pr_can_be_both_submitted_and_merged_same_window() -> None:
    """Independent lifecycle transitions, not mutually exclusive buckets -- a
    PR opened and merged within one short window is real signal, not a bug."""
    prs = [
        _pr(
            5,
            created_at="2026-07-01T08:00:00",
            merged_at="2026-07-01T09:00:00",
            closed_at="2026-07-01T09:00:00",
            state="MERGED",
        )
    ]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.submitted_count == 1
    assert result.merged_count == 1


def test_half_open_window_no_double_count_across_tick_boundary() -> None:
    """A PR merged exactly at a tick boundary (merged_at == until of tick 1 ==
    since of tick 2) must be counted in exactly one of the two adjacent,
    contiguous windows -- proves the half-open-interval dedup mechanism, not a
    per-event dedup store."""
    boundary = _dt("2026-07-02T00:00:00")
    prs = [_pr(6, created_at="2026-07-01T20:00:00", merged_at=boundary.isoformat(), closed_at=boundary.isoformat(), state="MERGED")]
    tick1 = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=boundary)
    tick2 = pr_lifecycle_delta(prs, since=boundary, until=_dt("2026-07-03T00:00:00"))
    assert tick1.merged_count + tick2.merged_count == 1
    assert tick2.merged_count == 1  # falls in tick2, since tick1's `until` is exclusive


def test_events_outside_window_are_excluded() -> None:
    prs = [
        _pr(7, created_at="2026-06-25T00:00:00", merged_at="2026-06-26T00:00:00", closed_at="2026-06-26T00:00:00", state="MERGED"),
        _pr(8, created_at="2026-07-10T00:00:00", state="OPEN"),
    ]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.submitted_count == 0
    assert result.merged_count == 0
    assert result.closed_without_merge_count == 0


def test_possibly_truncated_flag_true_when_fetch_capped_and_may_not_reach_window_start() -> None:
    """If the fetch appears capped (returned >= fetch_limit rows) and the
    oldest fetched updated_at doesn't reach back past `since`, the fetch may
    not have reached far enough back -- real events before that point could be
    silently missing. Must flag this rather than silently under-report."""
    prs = [_pr(9, created_at="2026-07-01T12:00:00", state="OPEN")]
    result = pr_lifecycle_delta(
        prs, since=_dt("2026-06-01T00:00:00"), until=_dt("2026-07-02T00:00:00"), fetch_limit=1
    )
    assert result.possibly_truncated is True


def test_possibly_truncated_flag_false_when_fetch_reaches_before_window() -> None:
    prs = [
        _pr(10, created_at="2026-05-01T00:00:00", state="MERGED", merged_at="2026-05-02T00:00:00", closed_at="2026-05-02T00:00:00"),
        _pr(11, created_at="2026-07-01T12:00:00", state="OPEN"),
    ]
    result = pr_lifecycle_delta(
        prs, since=_dt("2026-06-01T00:00:00"), until=_dt("2026-07-02T00:00:00"), fetch_limit=50
    )
    assert result.possibly_truncated is False


def test_possibly_truncated_flag_false_when_fetch_not_capped_even_if_all_events_recent() -> None:
    """Regression guard: without checking whether the fetch actually hit its
    limit, a window predating all real PR history would falsely flag
    truncation forever, since the oldest event would always look "too recent"
    relative to an old `since`. `len(prs) < fetch_limit` means gh returned
    everything available -- nothing was cut off, regardless of how recent it
    all is."""
    prs = [_pr(12, created_at="2026-07-01T12:00:00", state="OPEN")]
    result = pr_lifecycle_delta(
        prs, since=_dt("2020-01-01T00:00:00"), until=_dt("2026-07-02T00:00:00"), fetch_limit=200
    )
    assert result.possibly_truncated is False


def test_possibly_truncated_false_when_fetch_limit_not_provided() -> None:
    """Without a fetch_limit, there's no way to tell "capped" from "complete"
    -- must not guess, must default to not-flagged (same "no empty-shell
    cognition" avoidance as everywhere else in this program: an unknowable
    flag defaults to the less alarming state, not a fabricated one)."""
    prs = [_pr(13, created_at="2026-07-01T12:00:00", state="OPEN")]
    result = pr_lifecycle_delta(prs, since=_dt("2026-06-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.possibly_truncated is False


def test_old_pr_recently_merged_is_captured_via_updated_at_sort() -> None:
    """The bug this fix exists for: a PR created long before the window but
    merged inside it must still be counted -- proves the truncation check (and
    the real fetch ordering it models) is keyed off updated_at, which every
    lifecycle event bumps, not created_at, which only the submit event
    touches."""
    prs = [
        _pr(
            14,
            created_at="2025-01-01T00:00:00",  # ancient -- would rank last in a created-desc fetch
            merged_at="2026-07-01T12:00:00",
            closed_at="2026-07-01T12:00:00",
            state="MERGED",
        )
    ]
    result = pr_lifecycle_delta(
        prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"), fetch_limit=200
    )
    assert result.merged_count == 1
    # updated_at (== merged_at here) is inside the window, well past `since` --
    # not flagged, correctly, because this PR's real merge event was captured.
    assert result.possibly_truncated is False


def test_multiple_prs_span_all_three_buckets_in_one_call() -> None:
    prs = [
        _pr(20, created_at="2026-07-01T01:00:00", state="OPEN"),  # submitted only
        _pr(
            21,
            created_at="2026-06-20T00:00:00",
            merged_at="2026-07-01T02:00:00",
            closed_at="2026-07-01T02:00:00",
            state="MERGED",
        ),  # merged only
        _pr(
            22,
            created_at="2026-06-25T00:00:00",
            closed_at="2026-07-01T03:00:00",
            merged_at=None,
            state="CLOSED",
        ),  # closed-without-merge only
        _pr(
            23,
            created_at="2026-06-01T00:00:00",
            closed_at="2026-05-01T00:00:00",
            merged_at=None,
            state="CLOSED",
        ),  # entirely outside window -- contributes nothing
    ]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.submitted_numbers == (20,)
    assert result.merged_numbers == (21,)
    assert result.closed_without_merge_numbers == (22,)


def test_exceeds_max_digest_input_prs_cap_on_synthetic_window() -> None:
    """Regression guard mirroring the design spec's own acceptance check: a
    window with more than MAX_DIGEST_INPUT_PRS (8) real events must report the
    real count, not a value silently capped at 8 like
    trim_github_compactor_input()'s LLM-facing item list."""
    from orion.cognition.github_compactor.constants import MAX_DIGEST_INPUT_PRS

    prs = [
        _pr(100 + i, created_at="2026-07-01T00:00:00", merged_at=f"2026-07-01T{i:02d}:00:00", closed_at=f"2026-07-01T{i:02d}:00:00", state="MERGED")
        for i in range(MAX_DIGEST_INPUT_PRS + 3)
    ]
    result = pr_lifecycle_delta(prs, since=_dt("2026-07-01T00:00:00"), until=_dt("2026-07-02T00:00:00"))
    assert result.merged_count > MAX_DIGEST_INPUT_PRS


# -- fetch_recent_prs -----------------------------------------------------


def test_fetch_recent_prs_normalizes_gh_json(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = [
        {
            "number": 42,
            "createdAt": "2026-07-01T12:00:00Z",
            "mergedAt": "2026-07-01T13:00:00Z",
            "closedAt": "2026-07-01T13:00:00Z",
            "updatedAt": "2026-07-01T13:00:05Z",
            "state": "MERGED",
        },
        {
            "number": 43,
            "createdAt": "2026-07-01T14:00:00Z",
            "mergedAt": None,
            "closedAt": None,
            "updatedAt": "2026-07-01T14:00:00Z",
            "state": "OPEN",
        },
    ]

    def _fake_run(args, **kwargs):
        assert args[:3] == ["gh", "pr", "list"]
        assert "--search" in args and "sort:updated-desc" in args
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(raw), stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    prs = fetch_recent_prs("junebug-junie", "Orion-Sapienform", limit=50)

    assert len(prs) == 2
    assert prs[0]["number"] == 42
    assert prs[0]["created_at"] == _dt("2026-07-01T12:00:00")
    assert prs[0]["merged_at"] == _dt("2026-07-01T13:00:00")
    assert prs[0]["updated_at"] == _dt("2026-07-01T13:00:05")
    assert prs[1]["merged_at"] is None


def test_fetch_recent_prs_raises_with_stderr_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="fatal: not authenticated")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="not authenticated"):
        fetch_recent_prs("owner", "repo")


def test_fetch_recent_prs_raises_on_non_list_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """gh's own error payloads are JSON objects, not lists -- must surface as a
    clear error rather than silently returning an empty list, which would look
    identical to "genuinely zero PRs exist"."""
    def _fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps({"message": "API rate limit exceeded"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="non-list"):
        fetch_recent_prs("owner", "repo")
