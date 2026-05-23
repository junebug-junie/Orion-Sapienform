"""Repair pressure reducer tests — spec §14.2."""

from __future__ import annotations

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal.repair_pressure import appraise_repair_pressure
from orion.substrate.molecules import SubstrateMoleculeV1


def _obs(text: str, source_id: str = "msg-test") -> SubstrateMoleculeV1:
    return emit_observation(surface_text=text, source_id=source_id)


def test_high_repair_pressure_triggers_concrete_mode():
    # Each line below contributes at least one of the seven evidence kinds.
    # Across the window every kind fires at least once; the level formula
    # (spec §9.3) needs broad coverage to clear 0.75.
    molecules = [
        _obs("you gave me garbage directions"),  # trust_rupture + assistant_accountability_demand
        _obs("you keep making shit up — again"),  # repetition_failure + coherence_gap + assistant_accountability_demand
        _obs("this is becoming a swamp, doesn't converge"),  # coherence_gap
        _obs(
            "okay, arsonist POV only here: build me a design spec for Claude, "
            "not hand wavy, give me nuts and bolts"
        ),  # specificity_demand + operational_block + explicit_repair_command
    ]
    appraisal = appraise_repair_pressure(molecules, window_id="win-high")
    assert appraisal.dimensions["level"] >= 0.75, appraisal.dimensions
    assert appraisal.confidence >= 0.60, appraisal.confidence
    assert set(appraisal.causal_molecule_ids) >= {m.molecule_id for m in molecules}


def test_low_repair_pressure_neutral_chat():
    molecules = [_obs("what is the weather like?")]
    appraisal = appraise_repair_pressure(molecules, window_id="win-low")
    assert appraisal.dimensions["level"] <= 0.25, appraisal.dimensions
    assert appraisal.confidence <= 0.45, appraisal.confidence


def test_fail_closed_no_molecules():
    appraisal = appraise_repair_pressure([], window_id="win-empty")
    assert appraisal.dimensions["level"] == 0.0
    assert appraisal.confidence <= 0.25
    assert "no_repair_evidence" in appraisal.notes
    assert appraisal.evidence == []
    assert appraisal.causal_molecule_ids == []


def test_single_weak_evidence_caps_confidence():
    molecules = [_obs("again")]
    appraisal = appraise_repair_pressure(molecules, window_id="win-weak")
    assert appraisal.confidence <= 0.45, appraisal.confidence


def test_appraisal_records_all_required_dimensions():
    molecules = [_obs("not hand wavy, build me a design spec for Claude")]
    appraisal = appraise_repair_pressure(molecules, window_id="win-req")
    required = {
        "level",
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "assistant_accountability_demand",
        "confidence",
    }
    missing = required - appraisal.dimensions.keys()
    assert not missing, f"missing dimensions: {missing}"


def test_appraisal_kind_is_repair_pressure():
    appraisal = appraise_repair_pressure(
        [_obs("you gave me garbage directions")], window_id="win-kind"
    )
    assert appraisal.appraisal_kind == "repair_pressure"
