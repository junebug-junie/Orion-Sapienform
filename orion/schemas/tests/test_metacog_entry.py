from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.schemas.metacog_entry import (
    MetacogCausalDensity,
    MetacogEntryV1,
    MetacogProvenance,
    MetacogRealState,
    MetacogRepairEvidence,
    MetacogRepairPressure,
    MetacogWhatChanged,
)


def _minimal_entry(**overrides) -> MetacogEntryV1:
    defaults = dict(
        trigger_kind="relational",
        trigger_reason="repair_pressure:level=0.80:confidence=0.85",
        summary="A brief authored summary.",
        mantra="Stay grounded.",
        provenance=MetacogProvenance(
            source="cortex_exec.metacog_pipeline",
            produces="metacog_entry",
        ),
    )
    defaults.update(overrides)
    return MetacogEntryV1(**defaults)


def test_minimal_entry_has_defaults():
    entry = _minimal_entry()
    assert entry.event_id.startswith("metacog_")
    assert entry.timestamp
    assert entry.snapshot_kind == "baseline"
    assert entry.is_causally_dense is False
    assert entry.tags == []
    assert entry.epistemic_status == "observed"
    assert entry.visibility == "internal"
    assert entry.redaction_level == "low"
    assert entry.what_changed == MetacogWhatChanged()
    assert entry.state == MetacogRealState()
    assert entry.causal_density == MetacogCausalDensity()


def test_no_numeric_sisters_field_exists():
    """The whole point of this model: no self-report leg, no replacement."""
    assert "numeric_sisters" not in MetacogEntryV1.model_fields
    assert "numeric_sisters" not in MetacogRealState.model_fields


def test_snapshot_kind_rejects_free_text():
    """Learn from collapse_mirror's 38 distinct garbage snapshot_kind values --
    this field is a real Literal, not an unconstrained string."""
    with pytest.raises(ValidationError):
        _minimal_entry(snapshot_kind="strongly_tilted-positive")


def test_snapshot_kind_accepts_confirmed_dense():
    entry = _minimal_entry(snapshot_kind="confirmed_dense")
    assert entry.snapshot_kind == "confirmed_dense"


def test_trigger_kind_and_reason_required():
    with pytest.raises(ValidationError):
        MetacogEntryV1(
            summary="s",
            mantra="m",
            provenance=MetacogProvenance(source="x", produces="y"),
        )


def test_provenance_required():
    with pytest.raises(ValidationError):
        MetacogEntryV1(
            trigger_kind="relational",
            trigger_reason="r",
            summary="s",
            mantra="m",
        )


def test_state_real_state_repair_pressure_roundtrip():
    entry = _minimal_entry(
        state=MetacogRealState(
            biometrics={"status": "fresh"},
            turn_effect={"turn": {"valence": -0.4}},
            turn_effect_evidence={"phi_before": {"valence": 0.1}},
            substrate_eventfulness_score=0.42,
            substrate_eventfulness_reasons=["execution_pressure_spike"],
            llm_uncertainty={"logprob_summary": "stable"},
            reasoning_excerpt="A short excerpt of the reasoning trace.",
            repair_pressure=MetacogRepairPressure(
                level=0.8,
                level_label="HIGH",
                confidence=0.85,
                evidence=[
                    MetacogRepairEvidence(evidence_kind="trust_rupture", score=0.7, confidence=0.9)
                ],
                behavior_applied="repair_concrete",
            ),
        )
    )
    assert entry.state.substrate_eventfulness_score == 0.42
    assert entry.state.repair_pressure.level == 0.8
    assert entry.state.repair_pressure.evidence[0].evidence_kind == "trust_rupture"


def test_extra_fields_ignored():
    entry = MetacogEntryV1.model_validate(
        {
            "trigger_kind": "baseline",
            "trigger_reason": "heartbeat",
            "summary": "s",
            "mantra": "m",
            "provenance": {"source": "x", "produces": "y"},
            "numeric_sisters": {"valence": 0.9},  # must be silently dropped, not error
            "unexpected_field": "ignored",
        }
    )
    assert not hasattr(entry, "numeric_sisters")
    assert not hasattr(entry, "unexpected_field")


def test_provenance_impacts_defaults_empty_list():
    prov = MetacogProvenance(source="x", produces="y")
    assert prov.impacts == []
