"""RepairEvidenceV1 / RepairPressureAppraisalV1 model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.substrate.appraisal.models import (
    RepairEvidenceV1,
    RepairPressureAppraisalV1,
)


def _evidence(**overrides) -> RepairEvidenceV1:
    base = dict(
        evidence_id="ev_1",
        source_molecule_id="mol_abc",
        evidence_kind="specificity_demand",
        detector="phrase_match_v1",
        score=0.8,
        confidence=0.7,
        span="give me nuts and bolts",
        features={"phrase_strength": 0.9},
    )
    base.update(overrides)
    return RepairEvidenceV1(**base)


def test_evidence_accepts_known_kind():
    ev = _evidence()
    assert ev.evidence_kind == "specificity_demand"
    assert 0.0 <= ev.score <= 1.0
    assert 0.0 <= ev.confidence <= 1.0


def test_evidence_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        _evidence(evidence_kind="anger")


def test_evidence_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RepairEvidenceV1(
            evidence_id="ev_2",
            source_molecule_id="mol_a",
            evidence_kind="trust_rupture",
            detector="phrase_match_v1",
            score=0.5,
            confidence=0.5,
            mystery_field="nope",
        )


def test_evidence_clamps_via_validation():
    with pytest.raises(ValidationError):
        _evidence(score=1.5)
    with pytest.raises(ValidationError):
        _evidence(confidence=-0.1)


def test_appraisal_roundtrip():
    appraisal = RepairPressureAppraisalV1(
        appraisal_id="app_1",
        window_id="win_1",
        dimensions={
            "level": 0.6,
            "specificity_demand": 0.8,
            "trust_rupture": 0.0,
            "coherence_gap": 0.0,
            "repetition_failure": 0.0,
            "operational_block": 0.0,
            "explicit_repair_command": 0.0,
            "confidence": 0.5,
        },
        evidence=[_evidence()],
        causal_molecule_ids=["mol_abc"],
        confidence=0.5,
        summary="test",
    )
    assert appraisal.appraisal_kind == "repair_pressure"
    assert appraisal.dimensions["level"] == 0.6


def test_appraisal_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RepairPressureAppraisalV1(
            appraisal_id="app_2",
            window_id="win_2",
            dimensions={"level": 0.1, "confidence": 0.1},
            evidence=[],
            causal_molecule_ids=[],
            confidence=0.1,
            mystery="x",
        )
