from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.discussion_window.sql_fetch import select_turns


def _row(minutes_from_epoch: int, tag: str) -> dict:
    return {
        "created_at": datetime(2026, 7, 13, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes_from_epoch),
        "source": "hub_orion",
        "user_id": None,
        "correlation_id": tag,
        "prompt": f"prompt-{tag}",
        "response": f"response-{tag}",
    }


def test_contiguous_suffix_only_stops_at_gap_over_ninety_minutes() -> None:
    # Regression for the exact bug: a real conversation early in the window,
    # then a >90min idle gap, then a burst of unrelated activity right before
    # "now". contiguous_suffix_only=True must only keep the trailing cluster.
    rows = [
        _row(0, "early-1"),
        _row(5, "early-2"),
        _row(10, "early-3"),
        # >90 minute gap (5400s) here
        _row(600, "recent-1"),
        _row(605, "recent-2"),
    ]
    clustered, strategy = select_turns(rows, contiguous_suffix_only=True, max_turns=30)
    assert strategy == "time_bound_then_contiguous_suffix"
    assert [r["correlation_id"] for r in clustered] == ["recent-1", "recent-2"]


def test_contiguous_suffix_only_false_keeps_everything_across_the_gap() -> None:
    # "Compact the last N hours" must not lose the early conversation just
    # because a later idle period (or a burst of workflow-trigger noise, which
    # gets filtered separately by the compactor) broke contiguity.
    rows = [
        _row(0, "early-1"),
        _row(5, "early-2"),
        _row(10, "early-3"),
        _row(600, "recent-1"),
        _row(605, "recent-2"),
    ]
    clustered, strategy = select_turns(rows, contiguous_suffix_only=False, max_turns=30)
    assert strategy == "time_bound_recent_n"
    assert [r["correlation_id"] for r in clustered] == [
        "early-1",
        "early-2",
        "early-3",
        "recent-1",
        "recent-2",
    ]


def test_contiguous_suffix_only_false_respects_max_turns_cap() -> None:
    rows = [_row(i, f"t{i}") for i in range(5)]
    clustered, _ = select_turns(rows, contiguous_suffix_only=False, max_turns=3)
    assert [r["correlation_id"] for r in clustered] == ["t2", "t3", "t4"]


def test_empty_input_yields_empty_output_both_strategies() -> None:
    assert select_turns([], contiguous_suffix_only=True, max_turns=30) == ([], "time_bound_then_contiguous_suffix")
    assert select_turns([], contiguous_suffix_only=False, max_turns=30) == ([], "time_bound_recent_n")
