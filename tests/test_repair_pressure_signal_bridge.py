"""Repair pressure signal bridge tests."""

from __future__ import annotations

from orion.signals.registry import ORGAN_REGISTRY


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
