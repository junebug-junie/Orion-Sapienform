"""Definition-of-Done: full causal chain inspectable end-to-end. Spec §17."""

from __future__ import annotations

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal import (
    REPAIR_PRESSURE_DEBUG_KEY,
    appraise_repair_pressure,
    apply_repair_pressure_contract,
    extract_repair_evidence,
    repair_appraisal_to_signal,
    select_recent_chat_molecules,
)


def test_full_causal_chain_changes_next_response_contract():
    chat_lines = [
        "you gave me garbage directions",
        "you keep making shit up — again",
        "this is becoming a swamp, doesn't converge",
        "okay, arsonist POV only here: build me a design spec for Claude, not hand wavy, give me nuts and bolts",
    ]
    molecules = [
        emit_observation(surface_text=line, source_id="conv-DoD") for line in chat_lines
    ]

    window = select_recent_chat_molecules(molecules, source_id="conv-DoD")
    assert window, "windowing must keep at least the current turn"

    evidence = extract_repair_evidence(window)
    kinds = {e.evidence_kind for e in evidence}
    assert {"specificity_demand", "trust_rupture", "coherence_gap"} <= kinds

    appraisal = appraise_repair_pressure(window, window_id="win-DoD")
    assert appraisal.dimensions["level"] >= 0.75
    assert appraisal.confidence >= 0.60
    assert appraisal.causal_molecule_ids  # non-empty

    signal = repair_appraisal_to_signal(appraisal)
    assert signal.organ_id == "graph_cognition"
    assert signal.signal_kind == "repair_pressure"
    assert signal.causal_parents == appraisal.causal_molecule_ids

    contract = apply_repair_pressure_contract({"mode": "default"}, signal)
    assert contract["mode"] == "repair_concrete"
    debug = contract[REPAIR_PRESSURE_DEBUG_KEY]
    assert debug["mode_applied"] == "repair_concrete"
    assert set(debug["evidence_kinds"]) >= {
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
    }
    assert debug["causal_molecule_ids"] == appraisal.causal_molecule_ids


def test_neutral_turn_does_not_change_contract():
    molecules = [emit_observation(surface_text="what is the weather like?", source_id="conv-low")]
    window = select_recent_chat_molecules(molecules, source_id="conv-low")
    appraisal = appraise_repair_pressure(window, window_id="win-low")
    signal = repair_appraisal_to_signal(appraisal)
    contract = apply_repair_pressure_contract({"mode": "default"}, signal)
    assert contract["mode"] == "default"
    assert REPAIR_PRESSURE_DEBUG_KEY not in contract
