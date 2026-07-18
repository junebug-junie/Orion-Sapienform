from __future__ import annotations

import pytest

from orion.metacog.service import (
    IS_CAUSALLY_DENSE_THRESHOLD,
    compute_causal_density,
    turn_effect_severity,
)
from orion.schemas.metacog_entry import MetacogRealState, MetacogRepairEvidence, MetacogRepairPressure


def test_no_evidence_scores_ambient_zero():
    density = compute_causal_density(MetacogRealState())
    assert density.score == 0.0
    assert density.label == "ambient"
    assert density.rationale == "no_real_artifact_evidence"


def test_substrate_eventfulness_alone_scores_exactly_its_own_value():
    """Single-component blend normalizes by its own weight -- score should
    equal the raw value, not the value scaled down by the weight."""
    state = MetacogRealState(substrate_eventfulness_score=0.6)
    density = compute_causal_density(state)
    assert density.score == pytest.approx(0.6)
    assert density.label == "dense"


def test_high_repair_pressure_crosses_dense_threshold():
    state = MetacogRealState(
        repair_pressure=MetacogRepairPressure(
            level=0.9,
            level_label="HIGH",
            confidence=0.9,
            evidence=[MetacogRepairEvidence(evidence_kind="trust_rupture", score=0.8, confidence=0.9)],
        )
    )
    density = compute_causal_density(state)
    assert density.score == pytest.approx(0.81)
    assert density.score >= IS_CAUSALLY_DENSE_THRESHOLD
    assert "repair_pressure" in density.rationale


def test_low_confidence_repair_pressure_dampens_score():
    high_conf = compute_causal_density(
        MetacogRealState(
            repair_pressure=MetacogRepairPressure(level=0.9, level_label="HIGH", confidence=0.9)
        )
    )
    low_conf = compute_causal_density(
        MetacogRealState(
            repair_pressure=MetacogRepairPressure(level=0.9, level_label="HIGH", confidence=0.2)
        )
    )
    assert low_conf.score < high_conf.score


def test_turn_effect_severity_reads_largest_delta_magnitude():
    assert turn_effect_severity({"turn": {"valence": -0.8, "energy": 0.1}}) == pytest.approx(0.8)
    assert turn_effect_severity({"turn": {}}) is None
    assert turn_effect_severity({}) is None
    assert turn_effect_severity(None) is None


def test_turn_effect_severity_clamps_to_one():
    assert turn_effect_severity({"turn": {"valence": 5.0}}) == 1.0


def test_multiple_components_blend_together():
    state = MetacogRealState(
        substrate_eventfulness_score=0.9,
        turn_effect={"turn": {"coherence": -0.9}},
        repair_pressure=MetacogRepairPressure(level=0.9, level_label="HIGH", confidence=0.9),
    )
    density = compute_causal_density(state)
    # All three components near-maximal -> should land close to 1.0, well above threshold.
    assert density.score > IS_CAUSALLY_DENSE_THRESHOLD
    assert density.label in {"dense", "critical"}
    assert "repair_pressure" in density.rationale
    assert "substrate_eventfulness" in density.rationale
    assert "turn_effect_severity" in density.rationale


def test_score_never_exceeds_one_or_drops_below_zero():
    state = MetacogRealState(
        substrate_eventfulness_score=10.0,  # pathological input, should still clamp
        repair_pressure=MetacogRepairPressure(level=1.0, level_label="HIGH", confidence=1.0),
    )
    density = compute_causal_density(state)
    assert 0.0 <= density.score <= 1.0
