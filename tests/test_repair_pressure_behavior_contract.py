"""Behavior contract tests for repair pressure — spec §14.4 and §11.1."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.appraisal.contract import (
    REPAIR_PRESSURE_DEBUG_KEY,
    apply_repair_pressure_contract,
)


def _signal(level: float, confidence: float) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id="sig-test",
        organ_id="graph_cognition",
        organ_class=OrganClass.endogenous,
        signal_kind="repair_pressure",
        dimensions={
            "level": level,
            "specificity_demand": 0.9,
            "trust_rupture": 0.8,
            "coherence_gap": 0.7,
            "repetition_failure": 0.0,
            "operational_block": 0.6,
            "explicit_repair_command": 0.8,
            "assistant_accountability_demand": 0.0,
            "confidence": confidence,
        },
        causal_parents=["mol_a", "mol_b"],
        source_event_id="app-test",
        observed_at=now,
        emitted_at=now,
    )


def test_high_repair_pressure_forces_repair_concrete_mode():
    contract = apply_repair_pressure_contract({}, _signal(level=0.86, confidence=0.82))
    assert contract["mode"] == "repair_concrete"
    rules = contract["rules"]
    assert any("one concrete operational path" in r for r in rules)
    assert any("do not build" in r.lower() for r in rules)
    assert any("tests/acceptance checks" in r.lower() for r in rules)


def test_mid_repair_pressure_forces_concrete_bias():
    contract = apply_repair_pressure_contract({}, _signal(level=0.55, confidence=0.70))
    assert contract["mode"] == "concrete_bias"
    rules = contract["rules"]
    assert any("more specific" in r.lower() for r in rules)


def test_weak_repair_pressure_does_not_change_mode():
    contract = apply_repair_pressure_contract(
        {"mode": "default"}, _signal(level=0.30, confidence=0.90)
    )
    assert contract["mode"] == "default"
    assert "rules" not in contract or not contract["rules"]


def test_low_confidence_high_level_does_not_change_mode():
    contract = apply_repair_pressure_contract(
        {"mode": "default"}, _signal(level=0.86, confidence=0.40)
    )
    # level alone is not enough — confidence must also clear the bar.
    assert contract["mode"] == "default"


def test_none_signal_is_noop():
    contract = apply_repair_pressure_contract({"mode": "default"}, None)
    assert contract == {"mode": "default"}


def test_unrelated_signal_kind_is_ignored():
    sig = _signal(level=0.86, confidence=0.82)
    sig = sig.model_copy(update={"signal_kind": "goal_pressure"})
    contract = apply_repair_pressure_contract({"mode": "default"}, sig)
    assert contract["mode"] == "default"


def test_inspectable_debug_metadata_present_when_high():
    signal = _signal(level=0.86, confidence=0.82)
    contract = apply_repair_pressure_contract({}, signal)
    debug = contract[REPAIR_PRESSURE_DEBUG_KEY]
    assert debug["level"] == 0.86
    assert debug["confidence"] == 0.82
    assert debug["mode_applied"] == "repair_concrete"
    assert "specificity_demand" in debug["evidence_kinds"]
    assert debug["causal_molecule_ids"] == ["mol_a", "mol_b"]


def test_base_contract_is_not_mutated():
    base = {"mode": "default", "rules": ["keep"]}
    out = apply_repair_pressure_contract(base, _signal(level=0.86, confidence=0.82))
    assert base == {"mode": "default", "rules": ["keep"]}, base
    assert out is not base
