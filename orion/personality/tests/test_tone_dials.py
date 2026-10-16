"""Gate tests for the structural tone dials (pure, no I/O)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest  # noqa: E402

from orion.personality.tone_dials import DIAL_NAMES, ToneDials, measure_tone_dials  # noqa: E402

LISTY = """Great question! Here's a breakdown of what's going on.

## Overview

- **First**, the cache is cold.
- **Second**, the worker restarts.
- **Third**, the queue backs up.

## Next steps

1. Check the logs.
2. Restart the service.

Would you like me to dig deeper?"""

PLAIN = "The cache was cold after the restart. It warmed up in about a minute."


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t", "```\nprint(1)\n```"])
def test_empty_or_code_only_is_none(text):
    assert measure_tone_dials(text) is None


def test_listy_reply_outscores_plain_reply_on_assistant_shape():
    listy, plain = measure_tone_dials(LISTY), measure_tone_dials(PLAIN)
    assert isinstance(listy, ToneDials) and isinstance(plain, ToneDials)
    assert listy.heading_count == 2 and plain.heading_count == 0
    assert listy.bold_span_count == 3 and plain.bold_span_count == 0
    assert listy.list_line_ratio > 0.5 and plain.list_line_ratio == 0.0
    assert listy.question_count == 1 and plain.question_count == 0
    assert listy.opener_words > 0 and plain.opener_words == 0
    assert plain.sentence_count == 2
    assert plain.mean_sentence_words == pytest.approx(14 / 2)


def test_code_fences_are_ignored():
    fenced = PLAIN + "\n\n```python\n# heading?\n- not a list\nI me my **bold** —\n```\n"
    a, b = measure_tone_dials(PLAIN), measure_tone_dials(fenced)
    assert a == b


def test_unterminated_fence_runs_to_end():
    d = measure_tone_dials("Run this.\n```\n## not a heading\n- nope")
    assert d.heading_count == 0 and d.list_line_ratio == 0.0 and d.word_count == 2


def test_hyphenated_words_are_not_dashes():
    d = measure_tone_dials("A well-known, state-of-the-art, long-term plan.")
    assert d.em_dash_per_100_words == 0.0
    assert d.word_count == 5  # A, well-known, state-of-the-art, long-term, plan


def test_spaced_hyphen_and_em_dash_are_counted():
    d = measure_tone_dials("One thing — the cache - and another -- the queue.")
    # 8 words, 3 dashes
    assert d.word_count == 8
    assert d.em_dash_per_100_words == pytest.approx(100 * 3 / 8, abs=1e-3)


def test_list_marker_dash_is_not_a_dash():
    d = measure_tone_dials("- one\n- two")
    assert d.em_dash_per_100_words == 0.0 and d.list_line_ratio == 1.0


def test_first_person_is_closed_class_and_case_aware():
    d = measure_tone_dials("I think I'm right, and my view helps me. In Iowa it is fine.")
    # I, I'm, my, me -> 4 of 14 words; "In"/"Iowa"/"it" not counted
    assert d.word_count == 14
    assert d.first_person_singular_per_100_words == pytest.approx(100 * 4 / 14, abs=1e-3)
    curly = measure_tone_dials("I’ve seen it.")
    assert curly.first_person_singular_per_100_words == pytest.approx(100 / 3, abs=1e-3)


def test_single_paragraph_has_no_opener():
    assert measure_tone_dials("Just one paragraph here.\nStill the same one.").opener_words == 0


@pytest.mark.parametrize(
    "text",
    [
        "\u0000​﻿",
        "\U0001f600\U0001f600 ok?",
        "**unclosed bold",
        "### ",
        "\r\n\r\nWindows\r\nlines.\r\n",
        "日本語の文。こんにちは",
        "- \n* \n1) ",
        "a" * 20000,
        "?!?!...",
    ],
)
def test_odd_input_never_raises(text):
    d = measure_tone_dials(text)
    assert d is None or isinstance(d, ToneDials)


def test_dial_names_match_dict():
    assert tuple(measure_tone_dials(PLAIN).as_dict()) == DIAL_NAMES
