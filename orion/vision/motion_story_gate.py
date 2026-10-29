"""Juniper's spot-check gate for walkway motion stories, in one place.

Motion stories (``vision_motion_story``) stay Hub-only until Juniper has checked
them: at least ``GATE_MIN_CHECKED`` checked, at least ``GATE_MIN_RIGHT`` fully
right, and zero wrong names (2026-10-16). Two readers need the same answer:

- the Hub card (``services/orion-hub/scripts/motion_story_routes.py``) shows the
  tally and whether the gate passed;
- the walkway rollups (``services/orion-sql-writer/app/vision_rollup_loop.py``)
  include story sentences ONLY when it passed.

The tally counts one rules version at a time (the newest one with published
clean stories): a verdict on old rules never vouches for new ones. "Can't tell"
verdicts are never right and never wrong.

Pure: callers run their own SQL and pass the counts in. The Hub card still holds its own
copy (left in place so concurrent Hub work does not conflict);
``services/orion-hub/tests/test_walkway_rollup_routes.py`` fails if the two ever disagree.
"""

from __future__ import annotations

from typing import Any, Mapping

GATE_MIN_CHECKED = 50
GATE_MIN_RIGHT = 0.80
GATE_MAX_WRONG_NAMES = 0
VERDICTS = ("right", "wrong_name", "wrong_motion", "wrong_other")
CANT_TELL = "cant_tell"


def tally_from_counts(counts: Mapping[str, int], named_checked: int, rules_version: int | None) -> dict[str, Any]:
    """``counts`` maps verdict (or ``cant_tell``) -> stories with that verdict."""
    checked = sum(int(counts.get(v, 0) or 0) for v in VERDICTS)
    right = int(counts.get("right", 0) or 0)
    wrong_names = int(counts.get("wrong_name", 0) or 0)
    pct = right / checked if checked else None
    passed = (checked >= GATE_MIN_CHECKED and pct is not None and pct >= GATE_MIN_RIGHT
              and wrong_names <= GATE_MAX_WRONG_NAMES)
    return {
        "rules_version": rules_version,
        "checked": checked,
        "right": right,
        "wrong_name": wrong_names,
        "wrong_motion": int(counts.get("wrong_motion", 0) or 0),
        "wrong_other": int(counts.get("wrong_other", 0) or 0),
        "named_checked": int(named_checked or 0),
        "cant_tell": int(counts.get(CANT_TELL, 0) or 0),
        "pct_right": round(pct, 4) if pct is not None else None,
        "gate": {"min_checked": GATE_MIN_CHECKED, "min_right": GATE_MIN_RIGHT,
                 "max_wrong_names": GATE_MAX_WRONG_NAMES, "passed": passed},
    }


def gate_passed(tally: Mapping[str, Any] | None) -> bool:
    gate = (tally or {}).get("gate") or {}
    return bool(gate.get("passed"))
