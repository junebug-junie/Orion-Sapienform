from __future__ import annotations

from orion.cocreation.affective_signals import (
    AffectiveWordScores,
    aggregate_scores,
    count_swear_words,
    score_message,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("This is FINE, really.") == ["this", "is", "fine", "really"]


def test_count_swear_words_matches_curated_list_only() -> None:
    tokens = tokenize("this is fucking broken, damn it")
    assert count_swear_words(tokens) == 2


def test_score_message_swear_frequency_is_none_for_empty_text() -> None:
    scores = score_message("")
    assert scores.word_count == 0
    assert scores.swear_frequency is None


def test_score_message_computes_real_swear_frequency() -> None:
    scores = score_message("fuck fuck fuck this is fine")
    assert scores.word_count == 6
    assert scores.swear_count == 3
    assert scores.swear_frequency == 0.5


def test_score_message_typo_rate_flags_real_misspellings() -> None:
    scores = score_message("I definately think this is teh right approach")
    assert scores.checked_word_count > 0
    assert scores.unknown_word_count >= 1
    assert scores.typo_rate is not None
    assert scores.typo_rate > 0


def test_score_message_typo_rate_is_none_when_nothing_checkable() -> None:
    scores = score_message("ok PR SQL API")
    assert scores.checked_word_count == 0
    assert scores.typo_rate is None


def test_score_message_jargon_allowlist_does_not_inflate_typo_rate() -> None:
    scores = score_message(
        "orion falkordb postgres redis docker kubectl pytest github graphify substrate"
    )
    assert scores.checked_word_count == 0
    assert scores.typo_rate is None


def test_score_message_typo_rate_handles_curly_apostrophe_contractions() -> None:
    """iOS/macOS/many web fields auto-substitute a curly apostrophe (U+2019)
    for an ASCII one -- confirmed live 2026-08-11 against 15 real messages
    in this repo's own transcript corpus. Must not fragment "doesn’t"
    into "doesn" + "t" the way the ASCII-only fix originally missed."""
    scores = score_message("I don’t think this doesn’t work, it isn’t right")
    assert scores.typo_rate == 0.0


def test_count_swear_words_matches_possessive_or_contracted_form() -> None:
    assert count_swear_words(tokenize("this shit's broken")) == 1
    assert count_swear_words(tokenize("what the hell's going on")) == 1


def test_score_message_clean_technical_prose_reads_calm() -> None:
    scores = score_message(
        "Please check the pull request and merge it once the tests pass."
    )
    assert scores.swear_frequency == 0.0
    assert scores.typo_rate == 0.0


def test_aggregate_scores_sums_counts_not_averages_rates() -> None:
    short = AffectiveWordScores(
        word_count=1, swear_count=1, checked_word_count=0, unknown_word_count=0
    )
    long = AffectiveWordScores(
        word_count=199, swear_count=0, checked_word_count=50, unknown_word_count=0
    )
    combined = aggregate_scores([short, long])
    assert combined.word_count == 200
    assert combined.swear_count == 1
    assert combined.swear_frequency == 0.005
    assert combined.typo_rate == 0.0


def test_score_message_compute_typos_false_skips_spellcheck_entirely() -> None:
    """affective_state.py's producer never reads typo_rate -- must not pay
    for the spellcheck pass at all when it says so."""
    scores = score_message("I definately think this is teh right approach", compute_typos=False)
    assert scores.checked_word_count == 0
    assert scores.unknown_word_count == 0
    assert scores.typo_rate is None
    # swear_frequency (what this caller actually wants) is unaffected.
    assert scores.swear_frequency == 0.0


def test_aggregate_scores_of_empty_list_is_all_none_rates() -> None:
    combined = aggregate_scores([])
    assert combined.word_count == 0
    assert combined.swear_frequency is None
    assert combined.typo_rate is None
