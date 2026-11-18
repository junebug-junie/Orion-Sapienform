"""Pins run_reading_huge_page_replay.py's verdicts (seed c31227dc Stage 2, 2026-10-14).

Before: retries re-sent the identical prompt, no tool call was ever refused, and the
turn ran to the 3500 s wall with no draft. After: the repeat fetches and the late
fetches are refused, the last tool result the model waits for lands at 1728 s of a
3241 s budget, and a retry after a timeout is told not to fetch.
"""
from __future__ import annotations

from run_reading_huge_page_replay import replay_harness, replay_prompt


def test_prompt_size_was_not_the_cause_and_retry_now_differs():
    p = replay_prompt()
    assert p["retained_text_chars"] == 5544  # #2581's retained text: a 5.5k digest, already capped at 16k
    assert p["stored_prompt_chars_by_attempt"][2] < 25_000
    assert p["stored_attempts_2_and_3_identical"]  # before: the retry re-sent the doomed prompt
    assert p["retry_prompt_differs"] and p["new_retry_note_chars"] > 0
    assert p["new_note_chars"] < 500  # the fix adds one line, not a page


def test_before_nothing_stopped_the_turn():
    before = replay_harness(refetch_limit=0, fraction=0.0, budget=7200)
    assert before["blocked"] == []
    assert before["last_allowed_result_s"] == 3077 and before["turn_killed_at_s"] == 3251


def test_after_repeat_and_late_fetches_refused_with_time_left_to_answer():
    after = replay_harness(refetch_limit=1, fraction=0.5, budget=3241)
    by_id = {c["id"]: c["blocked_by"] for c in after["calls"]}
    assert by_id == {"t2": None, "t3": None, "t4": None, "t20": "refetch", "t21": "refetch",
                     "t22": None, "t99": "tool_window", "t100": "tool_window"}
    assert after["last_allowed_result_s"] == 1728
    assert after["budget_s"] - after["last_allowed_result_s"] > 1200  # >20 min left to write the JSON
