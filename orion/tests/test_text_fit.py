"""orion/text_fit.py: fit text into a budget without cutting a sentence."""

from __future__ import annotations

import random

from orion.text_fit import ELLIPSIS, fit_to_budget


def test_text_that_fits_is_unchanged() -> None:
    assert fit_to_budget("  I am a mesh of services.  ", 100) == "I am a mesh of services."


def test_an_over_budget_text_keeps_whole_sentences_and_marks_the_cut() -> None:
    text = "First sentence here. Second sentence is longer. Third one never fits in the budget."
    out = fit_to_budget(text, 60)
    assert out == "First sentence here. Second sentence is longer. " + ELLIPSIS
    assert len(out) <= 60


def test_closing_quotes_and_brackets_stay_with_their_sentence() -> None:
    text = 'I said "this holds." (Then I checked it.) And then a long tail that will not fit at all.'
    out = fit_to_budget(text, 50)
    assert out == 'I said "this holds." (Then I checked it.) ' + ELLIPSIS


def test_no_sentence_end_late_enough_falls_back_to_a_word_boundary() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve"
    out = fit_to_budget(text, 30)
    assert out.endswith(ELLIPSIS)
    assert not out[:-1].endswith(" ")
    assert out[:-1] in text  # whole words only
    assert text[len(out) - 1] == " "


def test_one_unbroken_token_is_hard_cut() -> None:
    assert fit_to_budget("x" * 50, 10) == "x" * 9 + ELLIPSIS


def test_never_longer_than_the_budget_and_never_mid_word() -> None:
    rng = random.Random(7)
    words = ["I", "am", "made", "of", "services,", "graphs", "and", "records.", "Unsure!", "why?"]
    for _ in range(400):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 120)))
        budget = rng.randint(5, 400)
        out = fit_to_budget(text, budget)
        assert len(out) <= budget
        if len(text) <= budget:
            assert out == text
            continue
        kept = out.removesuffix(ELLIPSIS).rstrip()
        # The kept text is a prefix of the original that ends at a word end.
        assert text.startswith(kept)
        window = text[: max(1, budget - 2)]
        if " " not in window[1:]:
            continue  # one unbroken token: a hard cut is the only cut
        assert text[len(kept)] == " " or kept.endswith(tuple(",;:")) or text[len(kept)] in ",;:"


def test_degenerate_budgets() -> None:
    assert fit_to_budget("abc", 0) == ""
    assert fit_to_budget("abcd", 1) == ELLIPSIS
    assert fit_to_budget("", 10) == ""
