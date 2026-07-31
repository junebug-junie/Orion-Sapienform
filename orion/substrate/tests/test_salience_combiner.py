"""Unit tests for `orion.substrate.attention.salience`'s GWT-coalition
Borda rank-aggregation (replaces the killed `SEED_WEIGHTS` linear combiner
-- see the module's own top docstring for the full kill/replace rationale).
"""

import pytest

from orion.schemas.attention_frame import AttentionSignalV1, SalienceFeaturesV1
from orion.substrate.attention.salience import (
    borda_coalition_salience,
    compute_evidence_features,
    evidence_breadth_for,
    evidence_strength_for,
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


def test_evidence_strength_is_max_salience_times_confidence():
    assert evidence_strength_for([_signal(0.9, 0.9)]) == pytest.approx(0.81)
    assert evidence_strength_for([]) == 0.0


def test_evidence_strength_is_monotonic():
    low = evidence_strength_for([_signal(0.2, 0.5)])
    high = evidence_strength_for([_signal(0.9, 0.9)])
    assert high > low


def test_evidence_breadth_rises_with_distinct_sources_and_refs():
    one = evidence_breadth_for([_signal(0.8, 0.8, source="current_turn", refs=["a"])])
    many = evidence_breadth_for(
        [
            _signal(0.8, 0.8, source="current_turn", refs=["a"]),
        ]
        + [_signal(0.8, 0.8, source="autonomy", refs=["b"])]
    )
    assert many > one


def test_compute_evidence_features_only_has_the_two_real_fields():
    feats = compute_evidence_features([_signal(0.9, 0.9)])
    assert isinstance(feats, SalienceFeaturesV1)
    dumped = feats.model_dump(mode="json")
    assert set(dumped) == {"schema_version", "evidence_strength", "evidence_breadth"}


# ---------------------------------------------------------------------------
# borda_coalition_salience: the real rank-aggregation over a competing set
# ---------------------------------------------------------------------------


def test_borda_coalition_salience_empty_set():
    assert borda_coalition_salience({}) == {}


def test_borda_coalition_salience_single_candidate_uses_raw_mean():
    """n == 1: nothing to rank against -- falls back to the loop's own raw
    mean(evidence_strength, evidence_breadth), not a general Borda formula."""
    feats = SalienceFeaturesV1(evidence_strength=0.8, evidence_breadth=0.4)
    result = borda_coalition_salience({"a": feats})
    assert result == {"a": pytest.approx(0.6)}


def test_borda_coalition_salience_discriminates_strong_vs_weak():
    strong = compute_evidence_features(
        [
            _signal(0.95, 0.9, source="current_turn"),
            _signal(0.8, 0.85, source="autonomy"),
            _signal(0.7, 0.8, source="concept_induction"),
        ]
    )
    weak = compute_evidence_features([_signal(0.35, 0.5, source="current_turn")])
    result = borda_coalition_salience({"strong": strong, "weak": weak})
    assert result["strong"] > result["weak"]
    # Two-candidate Borda: full-agreement winner gets both scorers' full
    # normalized-rank share -> exactly 1.0. Loser gets exactly 0.0.
    assert result["strong"] == pytest.approx(1.0)
    assert result["weak"] == pytest.approx(0.0)


def test_borda_coalition_salience_bounded_0_1():
    feats = {
        f"loop-{i}": compute_evidence_features([_signal(0.1 * i, 0.9)])
        for i in range(1, 6)
    }
    result = borda_coalition_salience(feats)
    assert set(result) == set(feats)
    for score in result.values():
        assert 0.0 <= score <= 1.0


def test_borda_coalition_salience_tie_shares_points():
    """Two candidates with identical evidence tie -- neither dominates."""
    a = compute_evidence_features([_signal(0.7, 0.8, source="current_turn")])
    b = compute_evidence_features([_signal(0.7, 0.8, source="autonomy")])
    result = borda_coalition_salience({"a": a, "b": b})
    assert result["a"] == pytest.approx(result["b"])
