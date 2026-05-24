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
    assert pressure_label(0.30) == "LOW"
    assert pressure_label(0.10) == "NONE"


def test_strength_label_buckets():
    assert strength_label(0.95) == "Very strong"
    assert strength_label(0.70) == "Strong"
    assert strength_label(0.50) == "Medium"
    assert strength_label(0.30) == "Low"
    assert strength_label(0.10) == "Very low"


def test_confidence_label_buckets():
    assert confidence_label(0.95) == "Very high"
    assert confidence_label(0.70) == "High"
    assert confidence_label(0.50) == "Medium"
    assert confidence_label(0.30) == "Low"
    assert confidence_label(0.10) == "Very low"


def test_kind_labels_translate_internal_enums():
    assert KIND_LABELS["specificity_demand"] == "Specificity demand"
    assert KIND_LABELS["trust_rupture"] == "Trust rupture"
    assert KIND_LABELS["repair_pressure"] == "Repair pressure"
    assert KIND_LABELS["repair_concrete"] == "Repair concrete mode"
    assert KIND_LABELS["normal_chat"] == "Normal chat"
