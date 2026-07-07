import pytest

from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1, SalienceFeaturesV1
from orion.substrate.attention.salience import (
    SalienceHistory,
    LinearSalienceCombiner,
    compute_salience,
    default_combiner,
)


def _signal(salience: float, confidence: float, source: str = "current_turn", refs=None):
    return AttentionSignalV1(
        signal_id=f"sig-{salience}-{confidence}-{source}",
        source=source,
        target_text="thing",
        signal_kind="test",
        salience=salience,
        confidence=confidence,
        evidence_refs=refs or ["r1"],
    )


def _loop(already_known: bool = False):
    return OpenLoopV1(id="open-loop-x", description="thing", already_known=already_known)


def test_score_is_bounded_and_deterministic():
    combiner = default_combiner()
    feats = SalienceFeaturesV1(evidence_strength=0.9, novelty_vs_known=0.8)
    a = combiner.score(feats)
    b = combiner.score(feats)
    assert a == b
    assert 0.0 <= a <= 1.0


def test_evidence_strength_is_monotonic():
    combiner = default_combiner()
    low = combiner.score(SalienceFeaturesV1(evidence_strength=0.2))
    high = combiner.score(SalienceFeaturesV1(evidence_strength=0.9))
    assert high > low


def test_habituation_strictly_lowers_salience():
    combiner = default_combiner()
    base = SalienceFeaturesV1(evidence_strength=0.8, novelty_vs_known=0.7)
    habituated = base.model_copy(update={"habituation": 0.9})
    assert combiner.score(habituated) < combiner.score(base)


def test_compute_salience_uses_signals():
    loop = _loop()
    strong, feats_strong = compute_salience(
        loop=loop, signals=[_signal(0.9, 0.9)], history=SalienceHistory(), now=None
    )
    weak, feats_weak = compute_salience(
        loop=loop, signals=[_signal(0.3, 0.5)], history=SalienceHistory(), now=None
    )
    assert strong > weak
    assert feats_strong.evidence_strength > feats_weak.evidence_strength


def test_already_known_lowers_novelty():
    _, feats_known = compute_salience(
        loop=_loop(already_known=True), signals=[_signal(0.9, 0.9)],
        history=SalienceHistory(), now=None,
    )
    _, feats_novel = compute_salience(
        loop=_loop(already_known=False), signals=[_signal(0.9, 0.9)],
        history=SalienceHistory(), now=None,
    )
    assert feats_novel.novelty_vs_known > feats_known.novelty_vs_known


def test_breadth_rises_with_distinct_detectors():
    _, one = compute_salience(
        loop=_loop(), signals=[_signal(0.8, 0.8, source="current_turn", refs=["a"])],
        history=SalienceHistory(), now=None,
    )
    _, many = compute_salience(
        loop=_loop(),
        signals=[
            _signal(0.8, 0.8, source="current_turn", refs=["a"]),
            _signal(0.8, 0.8, source="autonomy", refs=["b"]),
            _signal(0.8, 0.8, source="concept_induction", refs=["c"]),
        ],
        history=SalienceHistory(), now=None,
    )
    assert many.evidence_breadth > one.evidence_breadth
