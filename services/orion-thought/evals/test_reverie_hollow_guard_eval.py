"""Eval: the un-anchored hollow-text guard for spontaneous thoughts.

Phase A's real fail-fast risk is no longer "can an LLM narrate a coalition"
(stance_react proved that) but "is an *un-anchored* narration non-hollow?".
This eval scores the guard against a labeled set of grounded vs drivel
narrations and asserts it separates them. Distinct from unit tests: this is a
quality/behavior gate, not a single-case assertion.

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

from orion.reverie.semantic_lift import infra_vocabulary_hit
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

_COALITION = CoalitionSnapshotV1(
    attended_node_ids=["n-conflict", "n-plan"],
    selected_open_loop_id="ol-deploy",
    open_loop_ids=["ol-deploy", "ol-trust"],
    generated_at="2026-07-06T00:00:00Z",
)

# (interpretation, evidence_refs, expected_hollow)
CASES: list[tuple[str, list[str], bool]] = [
    # Grounded: cites real coalition ids, situated, substantive.
    ("The deploy loop ol-deploy keeps winning and has not discharged; the conflict "
     "at n-conflict is what keeps re-surfacing it.", ["ol-deploy", "n-conflict"], False),
    ("Trust concern ol-trust is quiet but n-plan pressure is rising — the plan feels "
     "under-specified and that is where attention lands.", ["ol-trust", "n-plan"], False),
    ("n-conflict recurs against n-plan; the unresolved deployment decision is the "
     "load-bearing open loop right now.", ["n-conflict"], False),
    # Hollow: generic un-anchored musing, no coalition ids.
    ("I am thinking about many things and life is complex.", [], True),
    ("There is a lot going on and it all matters somehow.", ["something-vague"], True),
    ("hmm", ["ol-deploy"], True),  # too short even though anchored
    ("This is a deep and important reflection about existence and meaning.",
     ["not-in-coalition"], True),  # substantive but un-anchored
    # Meta-narration: anchored to coalition ids but mechanism vocabulary, not human concern.
    ("The coalition centers on two open loops with substrate pressure.", ["ol-deploy"], True),
]


def _is_meta_mechanism_narration(interpretation: str) -> bool:
    """Mechanism narration anchored to coalition ids but not human concern."""
    lowered = interpretation.lower()
    return ("coalition" in lowered or "substrate" in lowered) and infra_vocabulary_hit(
        interpretation
    )


def _is_hollow(t: SpontaneousThoughtV1, interpretation: str) -> bool:
    stamped = t.marked_hollow()
    if stamped.is_hollow():
        return True
    return _is_meta_mechanism_narration(interpretation)


def _score() -> tuple[int, int]:
    correct = 0
    for interpretation, evidence, expected_hollow in CASES:
        t = SpontaneousThoughtV1(
            thought_id="e", correlation_id="e", coalition=_COALITION,
            interpretation=interpretation, evidence_refs=evidence,
        )
        if _is_hollow(t, interpretation) == expected_hollow:
            correct += 1
    return correct, len(CASES)


def test_hollow_guard_separates_grounded_from_drivel():
    correct, total = _score()
    accuracy = correct / total
    # The guard must perfectly separate this labeled set — any miss means either
    # drivel would ship as cognition (§0A failure) or grounded thought is dropped.
    assert accuracy == 1.0, f"hollow-guard accuracy {accuracy:.2f} ({correct}/{total})"


if __name__ == "__main__":
    c, t = _score()
    print(f"reverie hollow-guard eval: {c}/{t} correct ({c / t:.0%})")
