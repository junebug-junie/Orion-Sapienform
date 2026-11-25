"""The story verdict derived from Juniper's part answers (who-card redesign Patch C, 2026-11-24)."""

from __future__ import annotations

from orion.vision.motion_story_gate import (GATE_MAX_WRONG_NAMES, GATE_MIN_CHECKED, GATE_MIN_RIGHT, PARTS,
                                            derive_story_verdict, gate_verdicts_sql)


def test_nothing_is_derived_until_every_stated_part_is_answered() -> None:
    assert derive_story_verdict(["people", "direction"], {"people": "yes"}) is None
    assert derive_story_verdict([], {}) is None
    assert derive_story_verdict(["people"], {"people": "maybe"}) is None
    # A part the sentence does not state is never asked and never needed.
    assert derive_story_verdict(["people"], {"people": "yes"}) == "right"


def test_a_wrong_name_never_hides_behind_a_wrong_motion() -> None:
    every = ["people", "direction", "place"]
    assert derive_story_verdict(every, {"people": "yes", "direction": "yes", "place": "yes"}) == "right"
    assert derive_story_verdict(every, {"people": "no", "direction": "no", "place": "no"}) == "wrong_name"
    assert derive_story_verdict(every, {"people": "yes", "direction": "no", "place": "yes"}) == "wrong_motion"
    assert derive_story_verdict(every, {"people": "yes", "direction": "yes", "place": "no"}) == "wrong_motion"
    assert derive_story_verdict(every, {"people": "cant_tell", "direction": "no", "place": "yes"}) == "wrong_motion"
    assert derive_story_verdict(every, {"people": "cant_tell", "direction": "yes", "place": "yes"}) == "cant_tell"


def test_the_gate_numbers_do_not_move() -> None:
    assert (GATE_MIN_CHECKED, GATE_MIN_RIGHT, GATE_MAX_WRONG_NAMES) == (50, 0.80, 0)
    assert PARTS == ("people", "direction", "place")
    sql = gate_verdicts_sql("$1")
    assert "corrects_story_name" in sql and "s.rules_version = $1" in sql
