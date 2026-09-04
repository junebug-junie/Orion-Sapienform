from __future__ import annotations

from app.crawl.interest import InterestRule, rules_from_rows, score_candidate

_RULES = [
    InterestRule(keyword="gpu", weight=1.0),
    InterestRule(keyword="rtx", weight=1.0),
    InterestRule(keyword="ram", weight=1.0),
    InterestRule(keyword="network switch", weight=1.0),
    InterestRule(keyword="server", weight=1.0),
]


def test_keyword_hit_in_title_scores_and_explains():
    score, reasons = score_candidate(
        title="Asus Rog Nuc 2025 - RTX 5060 - 32 GB DDR5", description=None, price=2900.0, rules=_RULES
    )
    assert score > 0
    assert any("rtx" in r.lower() for r in reasons)
    # every reason is literal matched text, never a bare number
    assert all(isinstance(r, str) and len(r) > 3 for r in reasons)


def test_no_keyword_hit_scores_zero_with_no_reasons():
    score, reasons = score_candidate(title="Snow Storm Sleds", description=None, price=0.0, rules=_RULES)
    assert score == 0
    assert reasons == []


def test_price_band_bonus_only_applies_after_a_keyword_hit():
    # No keyword hit at all -> price band bonus must not fire on its own.
    score, reasons = score_candidate(title="Couch", description=None, price=500.0, rules=_RULES)
    assert score == 0
    assert reasons == []


def test_price_band_bonus_applies_on_top_of_a_keyword_hit():
    with_band, reasons_band = score_candidate(
        title="Dell server", description=None, price=500.0, rules=_RULES
    )
    without_band, reasons_no_band = score_candidate(
        title="Dell server", description=None, price=50000.0, rules=_RULES
    )
    assert with_band > without_band
    assert any("price" in r.lower() for r in reasons_band)
    assert not any("price" in r.lower() for r in reasons_no_band)


def test_word_boundary_prevents_substring_false_positive():
    # "ram" must not fire on "diagram" -- a real regression class for
    # short, common component abbreviations.
    score, reasons = score_candidate(
        title="Wiring diagram poster", description=None, price=None, rules=_RULES
    )
    assert score == 0
    assert reasons == []


def test_word_boundary_still_matches_ram_as_a_whole_word():
    score, reasons = score_candidate(title="32GB RAM kit", description=None, price=None, rules=_RULES)
    assert score > 0


def test_multiword_keyword_matches_as_a_phrase():
    score, reasons = score_candidate(
        title="Unmanaged network switch 24 port", description=None, price=None, rules=_RULES
    )
    assert score > 0
    assert any("network switch" in r.lower() for r in reasons)


def test_description_alone_can_trigger_a_keyword_hit():
    score, reasons = score_candidate(
        title="Great deal, message me",
        description="This is a used GPU in good condition.",
        price=None,
        rules=_RULES,
    )
    assert score > 0


def test_duplicate_keyword_occurrences_only_count_once():
    score, reasons = score_candidate(
        title="server server server", description=None, price=None, rules=_RULES
    )
    assert score == 1.0
    assert len(reasons) == 1


def test_rules_from_rows_skips_category_only_rows():
    rows = [
        {"keyword": None, "category_url": "https://classifieds.ksl.com/search/cat/Electronics", "weight": 0},
        {"keyword": "gpu", "category_url": None, "weight": 1.0},
    ]
    rules = rules_from_rows(rows)
    assert len(rules) == 1
    assert rules[0].keyword == "gpu"
