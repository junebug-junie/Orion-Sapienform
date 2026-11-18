"""Unnamed regulars without a bare living count (rules_version 5, 2026-11-03).

A regular is an unnamed appearance cluster seen on 2+ days. The matcher splits one person
into several clusters, so the old clause "68 unnamed regulars" (live day line 2026-11-02)
was one bare cluster count of every kind. Spec #2763 decision 5 (Juniper): no Orion-facing
line prints a bare people count.
"""

from __future__ import annotations

from orion.vision import walkway_rollup as wr

V4_LINE = (
    "Walkway Mon 2 Nov 2026: people: at least 2, at most 29 (the camera splits one person into several); "
    "dogs: at least 3, at most 11 (the camera splits one dog into several); 3 strollers (3 distinct), "
    "109 vehicles (70 distinct) passing; named: Bethany (17:22); 68 unnamed regulars (seen on another day "
    "this week too); busiest 17:00 (at most 31 passings, usual at most 54.3; people at least 2, at most 13)."
)


def test_regulars_phrase_counts_things_and_never_living_kinds() -> None:
    reg = {"count": 68, "by_kind": {"vehicle": 55, "person": 10, "dog": 2, "bicycle": 1}}
    assert wr.regulars_phrase(reg, "day") == (
        "unnamed regulars (seen on another day this week too): some people, some dogs, 1 bicycle, 55 vehicles"
    )
    assert wr.regulars_phrase(reg, "week").startswith("unnamed regulars (seen on 2+ days this week): ")
    assert wr.regulars_phrase({"count": 4}, "day") == "unnamed regulars (seen on another day this week too)"
    assert wr.regulars_phrase({"count": 0}, "day") == ""
    assert wr.regulars_phrase(None, "day") == ""


def test_old_stored_line_has_its_regulars_number_swapped_in_place() -> None:
    rollup = {"rules_version": 4, "grain": "day",
              "regulars": {"count": 68, "by_kind": {"vehicle": 60, "person": 8}}}
    shown = wr.display_line(V4_LINE, rollup)
    assert "68 unnamed regulars" not in shown
    assert "unnamed regulars (seen on another day this week too): some people, 60 vehicles; busiest" in shown
    # The rest of the stored line reads exactly as before.
    assert shown.replace("unnamed regulars (seen on another day this week too): some people, 60 vehicles",
                         "68 unnamed regulars (seen on another day this week too)") == V4_LINE


def test_new_rows_render_regulars_without_a_number_and_are_shown_as_stored() -> None:
    assert wr.RULES_VERSION >= wr.REGULARS_RULES_VERSION == 5
    stored = "Walkway Tue 3 Nov 2026: unnamed regulars (seen on another day this week too): some people."
    assert wr.display_line(stored, {"rules_version": 5, "grain": "day"}) == stored


def test_malformed_regulars_never_raise_into_readers() -> None:
    for reg in ({"count": 3, "by_kind": ["person"]}, {"count": 3, "by_kind": {"person": "x"}}, {"count": "x"}, "junk"):
        shown = wr.display_line(V4_LINE, {"rules_version": 4, "grain": "day", "regulars": reg})
        assert "68 unnamed regulars" not in shown
