from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from app.repair_pressure_metacog_gate import build_repair_pressure_metacog_trigger


def _appraisal(*, level=0.8, confidence=0.85, evidence=None, behavior_applied="repair_concrete") -> dict:
    return {
        "level": level,
        "level_label": "HIGH",
        "confidence": confidence,
        "evidence": evidence if evidence is not None else [
            {"evidence_kind": "trust_rupture", "score": 0.7, "confidence": 0.9}
        ],
        "behavior_applied": behavior_applied,
    }


def test_high_level_above_floors_fires_relational_trigger():
    trigger = build_repair_pressure_metacog_trigger(
        correlation_id="corr-1",
        appraisal=_appraisal(),
        zen_state="not_zen",
        pressure=0.4,
        recall_enabled=False,
        level_floor=0.5,
        confidence_floor=0.7,
    )
    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "relational"
    assert trigger.signal_refs == ["corr-1"]
    assert trigger.upstream["level"] == 0.8
    assert trigger.upstream["confidence"] == 0.85
    assert trigger.upstream["evidence"] == _appraisal()["evidence"]
    assert trigger.upstream["behavior_applied"] == "repair_concrete"


def test_low_level_below_floor_does_not_fire():
    trigger = build_repair_pressure_metacog_trigger(
        correlation_id="corr-2",
        appraisal=_appraisal(level=0.2),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        level_floor=0.5,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_low_confidence_below_floor_does_not_fire():
    trigger = build_repair_pressure_metacog_trigger(
        correlation_id="corr-3",
        appraisal=_appraisal(confidence=0.4),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        level_floor=0.5,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_missing_appraisal_does_not_fire():
    trigger = build_repair_pressure_metacog_trigger(
        correlation_id="corr-4",
        appraisal=None,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        level_floor=0.5,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_malformed_appraisal_missing_level_does_not_fire():
    trigger = build_repair_pressure_metacog_trigger(
        correlation_id="corr-5",
        appraisal={"confidence": 0.9},
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        level_floor=0.5,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_exactly_at_floors_fires():
    trigger = build_repair_pressure_metacog_trigger(
        correlation_id="corr-6",
        appraisal=_appraisal(level=0.5, confidence=0.7),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        level_floor=0.5,
        confidence_floor=0.7,
    )
    assert isinstance(trigger, MetacogTriggerV1)
