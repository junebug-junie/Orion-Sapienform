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


# Regressions from the 2026-10-16 review.

def test_fence_opened_mid_line_keeps_the_prose_after_it() -> None:
    d = measure_tone_dials("Here is code: ```python\nx=1\n```\nAfter one. After two. After three.")
    assert d is not None
    assert d.sentence_count == 4  # "Here is code:" + three trailing sentences
    assert d.word_count == 9  # x=1 is code, not words


def test_inline_triple_backtick_span_does_not_swallow_the_reply() -> None:
    d = measure_tone_dials("```ls``` is the command. Then more prose. And more.\nAnother line here.")
    assert d is not None
    assert d.sentence_count == 4


def test_longer_fence_is_not_closed_by_a_shorter_inner_fence() -> None:
    d = measure_tone_dials("````\n```\ninner\n```\n````\nOut.")
    assert d is not None
    assert d.word_count == 1


def test_list_markers_and_urls_do_not_inflate_word_count() -> None:
    assert measure_tone_dials("1. one\n2. two\n3. three").word_count == 3
    assert measure_tone_dials("see https://a.b.c/d-e/f_g?x=1&y=2 ok").word_count == 3


def test_question_wrapped_in_emphasis_counts() -> None:
    assert measure_tone_dials("**Want me to continue?**").question_count == 1
    d = measure_tone_dials("Is it *fine?* Yes.")
    assert d.question_count == 1
    assert d.sentence_count == 2


def test_leading_heading_is_not_a_preamble() -> None:
    assert measure_tone_dials("# Title\n\nBody text here.").opener_words == 0


# ", not X" contrast frame (the LLM voice tic Juniper flagged, 2026-11-18).

@pytest.mark.parametrize(
    "text, hits",
    [
        ("That's physiology, not hierarchy.", 1),
        ("It is a reported zero, not missing data.", 1),
        ("The cache is cold — not broken.", 1),
        ("The cache is cold —not broken.", 1),
        ("The cache is cold - not broken.", 1),
        ("The cache is cold -- not broken.", 1),
        ("The cache is cold – not broken.", 1),
        ("Read the log, Not the config.", 1),
        ("It's a bug, not just a quirk.", 1),
        ("It's merged, not yet deployed, not live.", 2),
        ("A signal, not noise. A choice — not a rule.", 2),
        ("- a reading, not a guess\n- a fact, not a vibe", 2),
        ("A concept node I can reach, not only a number in a description.", 1),
        ("It is cheap, *not* free.", 1),
        ("It is cheap, **not** free.", 1),
        ("It is cheap, _not_ free.", 1),
        ('It is cheap, not "free".', 1),
        ("It is cheap, not (free).", 1),
        ("The column is `status`, not `state`.", 1),
        ("*it landed*, not *it's right*.", 1),
        # Accepted on purpose (pinned): interjection commas and single-backtick
        # spans are prose to this dial.
        ("No, not yet.", 1),
        ("Use `a, not b` here.", 1),
        ("It matters — not just because it corrects the record, but because it holds.", 1),
    ],
)
def test_contrast_not_true_positives(text, hits):
    d = measure_tone_dials(text)
    assert d.contrast_not_count == hits
    assert d.contrast_not_per_100_words == pytest.approx(100 * hits / d.word_count, abs=1e-3)


@pytest.mark.parametrize(
    "text",
    [
        "I can't tell whether or not the worker restarted.",
        "Tell me if not, and I'll check again.",
        "Should I restart it or not?",
        "It works, but not on hecate.",
        "That is not the cache.",
        "It's the cache, nothing else.",
        "Three reasons, notably the heat.",
        "The fix, not only patches the parser but also adds a test.",
        "It costs power, not to mention the noise.",
        "Several causes, not least the heat in the closet.",
        "A well-known-not-great idea.",
        "Use the flag x-not y here.",
        "Ready, not.",
        "It helps, not only with speed but also with memory.",
        "Partial reply here.\n\n[Cut short - not a finished answer.]",
    ],
)
def test_contrast_not_false_positives(text):
    assert measure_tone_dials(text).contrast_not_count == 0


def test_contrast_not_inside_code_is_ignored():
    d = measure_tone_dials("Run it.\n```\nx = a, not b\n```\nInline ```y, not z``` too.")
    assert d.contrast_not_count == 0


def test_contrast_not_is_independent_of_dash_count():
    # A comma frame adds no dash; a dash frame adds both.
    comma = measure_tone_dials("A fact, not a vibe.")
    dash = measure_tone_dials("A fact — not a vibe.")
    assert comma.em_dash_per_100_words == 0.0 and comma.contrast_not_count == 1
    assert dash.em_dash_per_100_words > 0.0 and dash.contrast_not_count == 1
