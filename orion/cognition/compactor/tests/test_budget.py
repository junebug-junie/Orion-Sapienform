from __future__ import annotations

from orion.cognition.compactor.budget import fit_fields_within_budget


def test_fit_fields_within_budget_leaves_in_budget_values_untouched() -> None:
    # Exactly at the cap is in budget and must not be trimmed.
    fitted, trimmed = fit_fields_within_budget({"card_summary": ("abcde", 5)})
    assert trimmed == []
    assert fitted["card_summary"] == "abcde"


def test_fit_fields_within_budget_trims_over_budget_value_at_word_boundary() -> None:
    # 11 chars, cap 8. Reserving one char for the ellipsis gives a 7-char window
    # "one two"; the last whitespace in it is at index 3, and 3 > 8 * 0.5 is false,
    # so no boundary cut applies and the hand-computed result is "one two" + "…".
    fitted, trimmed = fit_fields_within_budget({"card_summary": ("one two ten", 8)})
    assert trimmed == ["card_summary"]
    assert fitted["card_summary"] == "one two\u2026"
    assert len(fitted["card_summary"]) == 8


def test_fit_fields_within_budget_never_exceeds_the_cap_it_was_given() -> None:
    # The ellipsis is paid for out of the budget, not added on top of it. An
    # unbroken token is the worst case: no whitespace to break on, so the result
    # is a hard cut plus the ellipsis and must still land exactly at the cap.
    fitted, _ = fit_fields_within_budget({"a": ("x" * 5000, 800)})
    assert len(fitted["a"]) == 800
    assert fitted["a"].endswith("\u2026")


def test_fit_fields_within_budget_reports_every_trimmed_field_sorted() -> None:
    fitted, trimmed = fit_fields_within_budget(
        {
            "card_summary": ("x" * 20, 5),
            "journal_title": ("ok", 50),
            "journal_body": ("y" * 20, 5),
        }
    )
    assert trimmed == ["card_summary", "journal_body"]
    assert fitted["journal_title"] == "ok"


def test_fit_fields_within_budget_handles_empty_value() -> None:
    fitted, trimmed = fit_fields_within_budget({"journal_title": ("", 10)})
    assert trimmed == []
    assert fitted["journal_title"] == ""


def test_fit_fields_within_budget_treats_none_as_empty() -> None:
    # The `value or ""` guard: both live callers already coerce, but len(None)
    # would raise here rather than degrade.
    fitted, trimmed = fit_fields_within_budget({"journal_title": (None, 10)})
    assert trimmed == []
    assert fitted["journal_title"] == ""


def test_fit_fields_within_budget_holds_the_cap_below_ellipsis_width() -> None:
    # A cap of 0 or 1 leaves no room to spend a character on the ellipsis, so the
    # result is a hard slice. Live caps are 800/120/4000, but the guarantee is
    # stated unconditionally and must actually hold unconditionally.
    for limit, expected in ((0, ""), (1, "h"), (2, "h…")):
        fitted, trimmed = fit_fields_within_budget({"a": ("hello world", limit)})
        assert fitted["a"] == expected
        assert len(fitted["a"]) <= limit
        assert trimmed == ["a"]


def test_fit_fields_within_budget_handles_whitespace_dominant_input() -> None:
    fitted, trimmed = fit_fields_within_budget({"a": (" " * 100, 5)})
    assert trimmed == ["a"]
    assert len(fitted["a"]) <= 5
