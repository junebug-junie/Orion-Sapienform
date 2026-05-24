from __future__ import annotations

from orion.substrate.appraisal.view_model import (
    KIND_LABELS,
    confidence_label,
    pressure_label,
    strength_label,
)


def test_pressure_label_buckets():
    assert pressure_label(0.90) == "HIGH"
    assert pressure_label(0.75) == "HIGH"
    assert pressure_label(0.50) == "MEDIUM"
    assert pressure_label(0.45) == "MEDIUM"
    assert pressure_label(0.30) == "LOW"
    assert pressure_label(0.25) == "LOW"
    assert pressure_label(0.10) == "NONE"


def test_strength_label_buckets():
    assert strength_label(0.95) == "Very strong"
    assert strength_label(0.85) == "Very strong"
    assert strength_label(0.70) == "Strong"
    assert strength_label(0.65) == "Strong"
    assert strength_label(0.50) == "Medium"
    assert strength_label(0.45) == "Medium"
    assert strength_label(0.30) == "Low"
    assert strength_label(0.25) == "Low"
    assert strength_label(0.10) == "Very low"


def test_confidence_label_buckets():
    assert confidence_label(0.95) == "Very high"
    assert confidence_label(0.85) == "Very high"
    assert confidence_label(0.70) == "High"
    assert confidence_label(0.65) == "High"
    assert confidence_label(0.50) == "Medium"
    assert confidence_label(0.45) == "Medium"
    assert confidence_label(0.30) == "Low"
    assert confidence_label(0.25) == "Low"
    assert confidence_label(0.10) == "Very low"


def test_kind_labels_translate_internal_enums():
    assert KIND_LABELS["specificity_demand"] == "Specificity demand"
    assert KIND_LABELS["trust_rupture"] == "Trust rupture"
    assert KIND_LABELS["repair_pressure"] == "Repair pressure"
    assert KIND_LABELS["repair_concrete"] == "Repair concrete mode"
    assert KIND_LABELS["normal_chat"] == "Normal chat"


from orion.substrate.appraisal.view_model import build_substrate_effect_view
from orion.substrate.appraisal import (
    appraise_repair_pressure,
    apply_repair_pressure_contract,
    extract_repair_evidence,
    repair_appraisal_to_signal,
    select_recent_chat_molecules,
)
from orion.mind.substrate_emit import emit_observation


def _run_pipeline(text, *, source_id: str = "src"):
    texts = [text] if isinstance(text, str) else list(text)
    mols = [emit_observation(surface_text=t, source_id=source_id) for t in texts]
    window = select_recent_chat_molecules(mols, source_id=source_id)
    appraisal = appraise_repair_pressure(window, window_id="w")
    signal = repair_appraisal_to_signal(appraisal)
    contract_before = {"mode": "default"}
    contract_after = apply_repair_pressure_contract(contract_before, signal)
    evidence = extract_repair_evidence(window)
    return appraisal, signal, evidence, contract_before, contract_after


def test_high_pressure_view_carries_repair_concrete_delta():
    appraisal, signal, evidence, before, after = _run_pipeline(
        [
            "you gave me garbage directions",
            "you keep making shit up — again",
            "this is becoming a swamp, doesn't converge",
            "okay, arsonist POV only here: build me a design spec for Claude, not hand wavy, give me nuts and bolts",
        ]
    )
    view = build_substrate_effect_view(
        turn_id="t-high",
        message_id="m-1",
        user_text="...",
        appraisal=appraisal,
        signal=signal,
        evidence=evidence,
        contract_before=before,
        contract_after=after,
    )
    assert view.outcome.level_label == "HIGH"
    assert view.behavior_delta is not None
    assert view.behavior_delta.changed is True
    assert view.behavior_delta.contract_after == "repair_concrete"
    assert any("Behavior changed" in step.title for step in view.causal_chain)
    assert view.evidence_cards, "evidence cards must populate"
    labels = [card.label for card in view.evidence_cards]
    assert any(label and label[0].isupper() for label in labels)
    primary = view.model_dump(exclude={"raw_debug"})
    assert primary["outcome"]["summary"]
    assert primary["behavior_delta"]["explanation"]


def test_medium_view_does_not_overstate():
    appraisal, signal, evidence, before, after = _run_pipeline(
        "build me a concrete design spec for the next step, focus on nuts and bolts"
    )
    view = build_substrate_effect_view(
        turn_id="t-med",
        message_id=None,
        user_text="...",
        appraisal=appraisal,
        signal=signal,
        evidence=evidence,
        contract_before=before,
        contract_after=after,
    )
    assert view.outcome.level_label in {"LOW", "MEDIUM", "HIGH"}
    if view.outcome.level_label == "MEDIUM":
        assert view.outcome.behavior_applied in {"concrete_bias", None}


def test_no_effect_returns_valid_empty_view():
    view = build_substrate_effect_view(
        turn_id="t-empty",
        message_id=None,
        user_text="",
        appraisal=None,
        signal=None,
        evidence=[],
        contract_before={"mode": "default"},
        contract_after={"mode": "default"},
    )
    assert view.outcome.appraisal_kind == "none"
    assert view.outcome.level == 0.0
    assert view.outcome.level_label == "NONE"
    assert view.outcome.behavior_applied is None
    assert "No substrate effect" in view.outcome.summary
    assert view.causal_chain == []
    assert view.evidence_cards == []
    assert view.molecule_summaries == []
