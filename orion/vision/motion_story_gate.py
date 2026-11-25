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

from typing import Any, Mapping, Optional, Sequence

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


# --- sentence parts (who-card Patch C, 2026-11-24) ------------------------------------------------------
# docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md section 2; Juniper's answer 3: people,
# direction, place (length left out). Which parts a story states is recorded when it is composed
# (orion-sql-writer vision_motion.sentence_parts -> vision_motion_story_parts), never parsed from prose.
PARTS = ("people", "direction", "place")
PART_VERDICTS = ("yes", "no", "cant_tell")
# The first "no" in this order names the story verdict: a wrong name never hides behind a wrong motion,
# because the gate's strictest rule is zero wrong names.
PART_FAILS = (("people", "wrong_name"), ("direction", "wrong_motion"), ("place", "wrong_motion"))


def derive_story_verdict(parts: Sequence[str], answers: Mapping[str, str]) -> Optional[str]:
    """The story verdict Juniper's part answers make, or None while any stated part is unanswered
    (a half-graded story stays in the unchecked queue and changes no count).

    All yes -> 'right'; any no -> the first of PART_FAILS that is no ('wrong_other' for a no on a
    part outside it); no no but a can't tell -> 'cant_tell'."""
    stated = [p for p in parts if p in PARTS]
    if not stated or any(answers.get(p) not in PART_VERDICTS for p in stated):
        return None
    said = {p: answers[p] for p in stated}
    if any(v == "no" for v in said.values()):
        for part, verdict in PART_FAILS:
            if said.get(part) == "no":
                return verdict
        return "wrong_other"
    if any(v == "cant_tell" for v in said.values()):
        return CANT_TELL
    return "right"


# --- the person-correction ledger, read at gate time (Orion's question 6) ------------------------------
# A story whose written name Juniper corrected on the story card ("someone else" on the named walker, or
# another name) counts as a WRONG NAME at gate time while that correction is her latest answer about that
# camera group (one vision_individual, the ask's source_ref; a linked person's answer writes one row per
# group, so undoing it on the person undoes it for every group) on that story -- whatever the story's
# verdict says (a later "right people? yes", a
# re-derived verdict, a button). Hub marks such an answer answer_detail.corrects_story_name = true when
# it writes it. Corrections before that mark carry their automatic 'wrong_name' verdict.
ACTIVE_CORRECTIONS_SQL = """
SELECT DISTINCT c.story_id FROM (
  SELECT answer_detail->>'story_id' AS story_id,
         coalesce(answer_detail @> '{"corrects_story_name": true}'::jsonb, false) AS corrects,
         row_number() OVER (PARTITION BY answer_detail->>'story_id', source_ref
                            ORDER BY answered_at DESC, created_at DESC, ask_id DESC) AS rn
  FROM orion_ask
  WHERE source_kind IN ('vision_individual', 'vision_story_person') AND status = 'answered'
    AND answer_detail ? 'story_id') c
WHERE c.rn = 1 AND c.corrects
"""


def gate_verdicts_sql(version_param: str) -> str:
    """Per gate verdict: (verdict, n, named) over published clean stories of one rules version, with
    every story under an active name correction counted as 'wrong_name'. ``version_param``: the
    driver's placeholder for the rules version (``$1`` for asyncpg, ``:v`` for SQLAlchemy)."""
    return (
        "WITH corrected AS (" + ACTIVE_CORRECTIONS_SQL + ") "
        "SELECT CASE WHEN c.story_id IS NOT NULL THEN 'wrong_name' WHEN v.cant_tell THEN 'cant_tell' "
        "ELSE v.verdict END AS verdict, count(*) AS n, count(*) FILTER (WHERE s.name_label IS NOT NULL) AS named "
        "FROM vision_motion_story s "
        "LEFT JOIN vision_motion_story_verdict v ON v.story_id = s.story_id "
        "LEFT JOIN corrected c ON c.story_id = s.story_id "
        f"WHERE s.published AND s.clean AND s.rules_version = {version_param} "
        "AND (v.story_id IS NOT NULL OR c.story_id IS NOT NULL) GROUP BY 1"
    )


def gate_passed(tally: Mapping[str, Any] | None) -> bool:
    gate = (tally or {}).get("gate") or {}
    return bool(gate.get("passed"))
