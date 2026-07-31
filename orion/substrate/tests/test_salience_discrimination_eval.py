"""Acceptance-gate eval: distinct coalitions must produce distinct salience.

Two plan-type loops with materially different evidence, built in the SAME
`build_open_loops()` call (the real competing set for one tick), must NOT
collapse to the same salience. This is the discrimination floor the killed
constant ladder failed (`score_loop()`'s legacy fallback pinned
`predictive_value=0.72` for both regardless of real evidence) and the
GWT-coalition Borda formula is required to clear.
"""

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.scoring import build_open_loops


def _sig(text: str, salience: float, confidence: float, source: str) -> AttentionSignalV1:
    return AttentionSignalV1(
        signal_id=f"sig-{text}-{source}",
        source=source,
        target_text=text,
        target_type_hint="plan",
        signal_kind="test",
        salience=salience,
        confidence=confidence,
        evidence_refs=[f"{text}-ref"],
    )


def test_distinct_coalitions_get_distinct_salience():
    strong_text = "open-loop-strong"
    weak_text = "open-loop-weak"
    signals = [
        _sig(strong_text, 0.95, 0.9, "current_turn"),
        _sig(strong_text, 0.8, 0.85, "autonomy"),
        _sig(strong_text, 0.7, 0.8, "concept_induction"),
        _sig(weak_text, 0.35, 0.5, "current_turn"),
    ]
    loops = build_open_loops(
        signals=signals,
        ctx={"user_message": f"{strong_text} {weak_text}"},
        inputs={},
        belief_lineage=[],
        direct_turn=False,
        generic_reversal=False,
        stale_thread_active=False,
        max_open=5,
    )
    by_desc = {loop.description: loop for loop in loops}
    strong = by_desc[strong_text]
    weak = by_desc[weak_text]

    # Discrimination floor: the two coalitions differ by a real margin.
    # merge_signals() keeps only ONE signal per target_text (the
    # highest-scoring), so "3 corroborating signals" collapses to the
    # single strongest one by the time build_open_loops sees it -- this is
    # existing, unchanged merge_signals() behavior, not new to this eval.
    assert strong.salience - weak.salience > 0.15, (
        f"salience did not discriminate: strong={strong.salience} weak={weak.salience}"
    )
