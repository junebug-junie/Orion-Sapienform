"""page:lately coverage eval scorer (issue #3463). Made-up rows only."""
from __future__ import annotations

from datetime import date, datetime, timezone

from orion.memory.pages.evals.lately_coverage import coverage


def _row(rid, day, purpose="happened", stakes="low", episode="ep1"):
    return {"record_id": rid, "occurred_at": datetime(2026, 11, day, 18, tzinfo=timezone.utc), "purpose": purpose,
            "stakes": stakes, "episode_id": episode}


ROWS = [_row("a", 17, "follow_up"), _row("b", 16, stakes="high", episode="ep2"), _row("c", 16, "about_juniper"),
        _row("d", 12, episode="ep3"), _row("old", 2, stakes="high", episode="ep9")]


def test_counts_only_rows_in_the_window_and_what_the_page_cites():
    refs = ["episode_memory:a", "episode_memory:b", "episode_memory:old", "chat_history_log:c1", "episode_memory:a"]
    out = coverage(refs, ROWS, since=date(2026, 11, 12))
    assert out["cited_refs"] == 4                                 # duplicates counted once
    assert out["cited_refs_from_window"] == 3                     # a, b and the chat turn; "old" is outside
    assert out["days_cited_of_days_with_memories"] == [2, 3]      # 11-17 and 11-16 of 11-17/16/12
    assert out["episodes_cited_of_episodes"] == [2, 3]
    assert out["high_stakes_cited_of_present"] == [1, 1]
    assert out["open_follow_ups_cited_of_present"] == [1, 1]
    assert out["about_juniper_cited_of_present"] == [0, 1]


def test_an_empty_page_covers_nothing():
    out = coverage([], ROWS, since=date(2026, 11, 12))
    assert out["cited_refs"] == 0 and out["days_cited_of_days_with_memories"] == [0, 3]
