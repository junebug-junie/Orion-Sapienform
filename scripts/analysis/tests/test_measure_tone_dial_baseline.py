"""Aggregation / separation logic of the tone dial baseline, on in-memory rows (no DB)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_tone_dial_baseline import (  # noqa: E402
    MIN_SEPARATION_N,
    build_report,
    cliffs_delta,
    dial_stats,
    percentile,
    render_summary,
)

LISTY = "Sure! Here's an overview.\n\n## Plan\n\n- **one**\n- **two**\n- **three**"
PLAIN = "The cache was cold. It warmed up."


def test_percentile_interpolates():
    assert percentile([], 0.5) is None
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.9) == 5


def test_cliffs_delta_extremes():
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
    assert cliffs_delta([], [1]) is None


def test_dial_stats_flags_degenerate():
    assert dial_stats([0, 0, 0])["degenerate"] == "always_zero"
    assert dial_stats([2, 2])["degenerate"] == "constant"
    assert dial_stats([0] * 19 + [1])["degenerate"] == "near_always_zero"
    assert dial_stats([0, 1, 2])["degenerate"] is None
    assert dial_stats([])["degenerate"] == "no_data"


def test_outreach_is_kept_separate_and_empty_responses_skipped():
    history = [
        {"correlation_id": "c1", "prompt": "hi", "response": PLAIN},
        {"correlation_id": "c2", "prompt": "", "response": LISTY},
        {"correlation_id": "c3", "prompt": None, "response": LISTY},
        {"correlation_id": "c4", "prompt": "hi", "response": "```\ncode\n```"},
    ]
    r = build_report(history, [], window_days=7)
    assert r["counts"]["chat_replies"] == 1
    assert r["counts"]["outreach"] == 2
    assert r["counts"]["skipped_unmeasurable"] == 1
    assert r["groups"]["outreach"]["heading_count"]["median"] == 1
    assert r["groups"]["chat_replies"]["heading_count"]["median"] == 0


def _history(n_listy: int, n_plain: int):
    rows = [{"correlation_id": f"l{i}", "prompt": "q", "response": LISTY} for i in range(n_listy)]
    rows += [{"correlation_id": f"p{i}", "prompt": "q", "response": PLAIN} for i in range(n_plain)]
    return rows


def test_small_n_reports_insufficient_not_a_number():
    feedback = [
        {"target_correlation_id": "p0", "feedback_value": "up", "categories": ["good_tone"]},
        {"target_correlation_id": "l0", "feedback_value": "down", "categories": ["bad_tone"]},
        {"target_correlation_id": "missing", "feedback_value": "down", "categories": ["bad_tone"]},
    ]
    r = build_report(_history(1, 1), feedback, window_days=7)
    sep = r["separation"]["good_tone_vs_bad_tone"]
    assert sep["n_good"] == 1 and sep["n_bad"] == 1 and not sep["sufficient_n"]
    assert sep["dials"]["heading_count"]["cliffs_delta"] == "insufficient n"
    assert sep["dials"]["heading_count"]["good_median"] == 0
    assert sep["dials"]["heading_count"]["bad_median"] == 1
    assert r["counts"]["feedback_joined_in_window"] == 2
    assert "insufficient n" in render_summary(r)


def test_sufficient_n_separates_and_incorrect_tone_counts_as_bad():
    n = MIN_SEPARATION_N
    feedback = [{"target_correlation_id": f"p{i}", "feedback_value": "up", "categories": ["good_tone"]} for i in range(n)]
    feedback += [
        {"target_correlation_id": f"l{i}", "feedback_value": "down",
         "categories": ["incorrect_tone" if i % 2 else "bad_tone"]}
        for i in range(n)
    ]
    r = build_report(_history(n, n), feedback, window_days=7)
    sep = r["separation"]["good_tone_vs_bad_tone"]
    assert sep["sufficient_n"] and sep["n_bad"] == n
    # Good (plain) replies have fewer headings than bad (listy) ones -> delta -1.
    assert sep["dials"]["heading_count"]["cliffs_delta"] == -1.0
    assert r["separation"]["thumbs_up_vs_down"]["dials"]["bold_span_count"]["cliffs_delta"] == -1.0
