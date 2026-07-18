"""_extract_entities: regression tests for a double-backslash regex bug found
in code review (2026-07-19, during the entity-relatedness fusion-boost PR).
A literal double-backslash inside an r-string (e.g. r"...\\s...") matches a
literal backslash character, not whitespace/a dot -- this silently broke
multi-word entity spans and dotted-identifier matching, and excluded
all-caps acronyms, for as long as this function has existed. Never
previously tested."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.worker import _extract_entities


def test_multi_word_entity_spans_merge_not_split() -> None:
    """Regression: the double-backslash bug meant '\\s+' matched a literal
    backslash, never whitespace, so this always came back as ["New",
    "York"] separately -- never merged."""
    out = _extract_entities("Tell me about New York and Elon Musk please.")
    assert "New York" in out
    assert "Elon Musk" in out
    assert "New" not in out
    assert "York" not in out


def test_all_caps_acronym_matches() -> None:
    """Regression: [A-Z][a-z]+ requires lowercase letters after the first
    capital, so an all-caps word like 'NVIDIA' matched zero times."""
    out = _extract_entities("Tell me about NVIDIA and Nvidia please.")
    assert "NVIDIA" in out
    assert "Nvidia" in out


def test_dotted_identifiers_still_match() -> None:
    """Regression: the double-backslash bug also broke the dotted-identifier
    pattern (r"...\\.[...\\.]+") the same way -- 'settings.py'/'example.com'
    never matched at all before this fix."""
    out = _extract_entities("check services/orion-recall/settings.py and example.com")
    assert "settings.py" in out
    assert "example.com" in out


def test_uuid_pattern_unaffected() -> None:
    out = _extract_entities("turn f223ea80-1de3-46f8-8015-ffef48f5992b happened")
    assert "f223ea80-1de3-46f8-8015-ffef48f5992b" in out


def test_empty_text_returns_empty_list() -> None:
    assert _extract_entities("") == []


def test_no_capitalized_words_returns_empty_for_that_pattern() -> None:
    out = _extract_entities("what's the weather like today")
    assert out == []


def test_bare_single_capital_letter_does_not_match() -> None:
    """Regression: an earlier version of this fix used [a-zA-Z]* (zero-or-
    more) instead of [a-zA-Z]+ (one-or-more), which also matched bare
    single capital letters like the pronoun "I" -- a real regression on the
    already-live sql_timeline.py::fetch_related_by_entities call site,
    which would have built an unbounded ILIKE '%I%' from it."""
    out = _extract_entities("I like pizza")
    assert "I" not in out
    assert out == []


def test_multi_word_merge_capped_at_two_words() -> None:
    """Regression: an unbounded merge group (`*` instead of `?`) would
    collapse a long Title-Case sentence into one giant string that then
    matches nothing real in Falkor or Postgres. Capped at one extra word,
    matching app/recall_v2.py::_extract_entities' own existing convention."""
    out = _extract_entities("Please Review The New Recall Entity Extraction Fix")
    assert all(len(e.split()) <= 2 for e in out)
    assert "Please Review The New" not in out
