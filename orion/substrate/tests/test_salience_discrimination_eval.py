"""Recommendation #1: distinct coalitions must produce distinct salience.

Two plan-type loops with materially different evidence should NOT collapse to
the same salience. Today's constant ladder pins them (predictive=0.72 for both).
The v2 combiner discriminates. This eval is the acceptance gate for the flip.
"""

from orion.substrate.attention.salience import SalienceHistory, compute_salience
from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1


def _plan_loop(loop_id: str) -> OpenLoopV1:
    return OpenLoopV1(id=loop_id, target_type="plan", description=loop_id)


def _sig(loop_id: str, salience: float, confidence: float, source: str) -> AttentionSignalV1:
    return AttentionSignalV1(
        signal_id=f"sig-{loop_id}-{source}",
        source=source,
        target_text=loop_id,
        target_type_hint="plan",
        signal_kind="test",
        salience=salience,
        confidence=confidence,
        evidence_refs=[f"{loop_id}-ref"],
    )


def test_distinct_coalitions_get_distinct_salience():
    strong_loop = _plan_loop("open-loop-strong")
    weak_loop = _plan_loop("open-loop-weak")

    strong, _ = compute_salience(
        loop=strong_loop,
        signals=[
            _sig("open-loop-strong", 0.95, 0.9, "current_turn"),
            _sig("open-loop-strong", 0.8, 0.85, "autonomy"),
            _sig("open-loop-strong", 0.7, 0.8, "concept_induction"),
        ],
        history=SalienceHistory(),
        apply_habituation=False,
    )
    weak, _ = compute_salience(
        loop=weak_loop,
        signals=[_sig("open-loop-weak", 0.35, 0.5, "current_turn")],
        history=SalienceHistory(),
        apply_habituation=False,
    )

    # Discrimination floor: the two coalitions differ by a real margin.
    assert strong - weak > 0.15, f"salience did not discriminate: {strong=} {weak=}"


def test_legacy_constant_ladder_fails_discrimination():
    """Evidence the old path was broken: two plan loops collapse to one score."""
    from orion.substrate.attention.scoring import score_loop

    a = OpenLoopV1(id="a", target_type="plan", description="a", predictive_value=0.72,
                   novelty=0.35, concept_value=0.55)
    b = OpenLoopV1(id="b", target_type="plan", description="b", predictive_value=0.72,
                   novelty=0.35, concept_value=0.55)
    # Legacy score_loop reads only the constant fields → identical inputs → identical score.
    assert abs(score_loop(a) - score_loop(b)) < 1e-9
