"""Repair pressure signal bridge tests."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.mind.substrate_emit import emit_observation
from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.registry import ORGAN_REGISTRY
from orion.substrate.appraisal.repair_pressure import appraise_repair_pressure
from orion.substrate.appraisal.signal_bridge import repair_appraisal_to_signal


def test_graph_cognition_registers_repair_pressure_signal_kind():
    entry = ORGAN_REGISTRY["graph_cognition"]
    assert "repair_pressure" in entry.signal_kinds, (
        "graph_cognition organ must register 'repair_pressure' so the appraisal "
        "signal bridge emits a canonical signal kind, not an ad-hoc one."
    )


def test_graph_cognition_canonical_dimensions_cover_repair_pressure():
    entry = ORGAN_REGISTRY["graph_cognition"]
    required = {
        "level",
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "confidence",
    }
    missing = required - set(entry.canonical_dimensions)
    assert not missing, f"graph_cognition canonical_dimensions missing: {missing}"


def _high_appraisal():
    molecules = [
        emit_observation(surface_text="you gave me garbage directions", source_id="m1"),
        emit_observation(surface_text="you keep making shit up — again", source_id="m1"),
        emit_observation(surface_text="this is becoming a swamp, doesn't converge", source_id="m1"),
        emit_observation(
            surface_text=(
                "okay, arsonist POV only here: build me a design spec for Claude, "
                "not hand wavy, give me nuts and bolts"
            ),
            source_id="m1",
        ),
    ]
    return appraise_repair_pressure(molecules, window_id="win-bridge")


def test_bridge_emits_graph_cognition_repair_pressure_signal():
    appraisal = _high_appraisal()
    signal = repair_appraisal_to_signal(appraisal)
    assert isinstance(signal, OrionSignalV1)
    assert signal.organ_id == "graph_cognition"
    assert signal.signal_kind == "repair_pressure"
    assert signal.organ_class == OrganClass.endogenous


def test_bridge_dimensions_match_appraisal():
    appraisal = _high_appraisal()
    signal = repair_appraisal_to_signal(appraisal)
    for key in (
        "level",
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "confidence",
    ):
        assert signal.dimensions[key] == appraisal.dimensions[key], key


def test_bridge_carries_causal_parents():
    appraisal = _high_appraisal()
    signal = repair_appraisal_to_signal(appraisal)
    assert signal.causal_parents == appraisal.causal_molecule_ids
    assert signal.source_event_id == appraisal.appraisal_id


def test_bridge_uses_deterministic_signal_id():
    appraisal = _high_appraisal()
    s1 = repair_appraisal_to_signal(appraisal)
    s2 = repair_appraisal_to_signal(appraisal)
    assert s1.signal_id == s2.signal_id


def test_bridge_observed_at_override():
    appraisal = _high_appraisal()
    when = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    signal = repair_appraisal_to_signal(appraisal, observed_at=when)
    assert signal.observed_at == when


def test_bridge_caps_notes():
    appraisal = _high_appraisal()
    appraisal.notes = [f"note_{i}" for i in range(10)]
    signal = repair_appraisal_to_signal(appraisal)
    assert len(signal.notes) <= 5
