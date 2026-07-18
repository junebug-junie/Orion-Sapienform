from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from app.relational_metacog_gate import build_relational_metacog_trigger


def _appraisal(*, shift_kind, novelty=0.9, confidence=0.85, status="ok") -> dict:
    return {
        "turn_change_status": status,
        "shift_kind": shift_kind,
        "novelty_score": novelty,
        "confidence": confidence,
    }


def test_repair_shift_above_confidence_floor_fires_relational_trigger():
    trigger = build_relational_metacog_trigger(
        correlation_id="corr-1",
        turn_change_appraisal=_appraisal(shift_kind="REPAIR"),
        zen_state="not_zen",
        pressure=0.4,
        recall_enabled=False,
        confidence_floor=0.7,
    )
    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "relational"
    assert trigger.signal_refs == ["corr-1"]
    assert trigger.upstream["shift_kind"] == "REPAIR"
    assert trigger.upstream["confidence"] == 0.85


def test_topic_shift_above_confidence_floor_fires_relational_trigger():
    trigger = build_relational_metacog_trigger(
        correlation_id="corr-2",
        turn_change_appraisal=_appraisal(shift_kind="TOPIC"),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        confidence_floor=0.7,
    )
    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "relational"


def test_none_shift_does_not_fire():
    trigger = build_relational_metacog_trigger(
        correlation_id="corr-3",
        turn_change_appraisal=_appraisal(shift_kind="NONE"),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_stance_shift_does_not_fire_yet():
    """STANCE is explicitly left undecided -- see the redesign doc's 'Still open' section."""
    trigger = build_relational_metacog_trigger(
        correlation_id="corr-4",
        turn_change_appraisal=_appraisal(shift_kind="STANCE"),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_low_confidence_below_floor_does_not_fire():
    trigger = build_relational_metacog_trigger(
        correlation_id="corr-5",
        turn_change_appraisal=_appraisal(shift_kind="REPAIR", confidence=0.4),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_degraded_status_does_not_fire():
    trigger = build_relational_metacog_trigger(
        correlation_id="corr-6",
        turn_change_appraisal=_appraisal(shift_kind="REPAIR", status="degraded"),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        confidence_floor=0.7,
    )
    assert trigger is None


def test_missing_appraisal_does_not_fire():
    trigger = build_relational_metacog_trigger(
        correlation_id="corr-7",
        turn_change_appraisal=None,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        confidence_floor=0.7,
    )
    assert trigger is None
