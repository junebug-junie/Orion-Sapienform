from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.drives import DriveAuditV1, DriveStateV1, TensionEventV1
from orion.spark.concept_induction.audit import build_drive_audit
from orion.spark.concept_induction.drive_attribution import (
    DRIVE_KEYS,
    compute_tick_attribution,
    dominant_drive_from_attribution,
    select_lead_tension,
)


def _gap_tension() -> TensionEventV1:
    return TensionEventV1.model_validate(
        {
            "artifact_id": "tension-gap-gpu",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "substrate.world_coverage_gap",
            "magnitude": 0.65,
            "drive_impacts": {"predictive": 0.15},
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


def _contradiction_tension() -> TensionEventV1:
    return TensionEventV1.model_validate(
        {
            "artifact_id": "tension-coh",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "tension.contradiction.v1",
            "magnitude": 0.5,
            "drive_impacts": {"coherence": 0.4, "autonomy": 0.4},
            "provenance": {"intake_channel": "orion:metacognition:tick"},
        }
    )


def test_attribution_gpu_gap_predictive_dominant() -> None:
    """Spec acceptance 1: saturated pressures irrelevant; gap → predictive."""
    gap = _gap_tension()
    attribution = compute_tick_attribution(
        [gap],
        metabolism_deltas={"predictive": 0.15},
    )
    dominant = dominant_drive_from_attribution(attribution, lead_tension=gap)
    assert dominant == "predictive"
    assert attribution["predictive"] > attribution.get("autonomy", 0.0)


def test_attribution_no_alphabetical_autonomy() -> None:
    """Spec acceptance 2: equal attribution ties break on lead tension kind, not 'autonomy'."""
    tension = _contradiction_tension()
    attribution = {key: 0.1 for key in DRIVE_KEYS}
    dominant = dominant_drive_from_attribution(attribution, lead_tension=tension)
    assert dominant == "coherence"
    assert dominant != "autonomy"


def test_metabolism_deltas_merge_into_attribution() -> None:
    """Spec acceptance 3: drive_deltas alone shift dominance when tensions empty."""
    attribution = compute_tick_attribution([], metabolism_deltas={"predictive": 0.2})
    dominant = dominant_drive_from_attribution(attribution, lead_tension=None)
    assert dominant == "predictive"
    assert attribution["predictive"] == 0.2


def test_select_lead_tension_prefers_gap_on_magnitude_tie() -> None:
    gap = _gap_tension()
    other = _contradiction_tension().model_copy(update={"magnitude": gap.magnitude})
    lead = select_lead_tension([other, gap])
    assert lead is not None
    assert lead.kind == "substrate.world_coverage_gap"


def test_dominant_drive_none_when_all_zero() -> None:
    attribution = {key: 0.0 for key in DRIVE_KEYS}
    assert dominant_drive_from_attribution(attribution, lead_tension=None) is None


def test_dominant_drive_alphabetical_fallback_without_lead() -> None:
    attribution = {key: 0.2 for key in DRIVE_KEYS}
    dominant = dominant_drive_from_attribution(attribution, lead_tension=None)
    assert dominant == "autonomy"


def test_select_lead_tension_empty() -> None:
    assert select_lead_tension([]) is None


def test_select_lead_tension_highest_magnitude() -> None:
    gap = _gap_tension()
    other = _contradiction_tension()
    lead = select_lead_tension([other, gap])
    assert lead is not None
    assert lead.kind == gap.kind
    assert lead.magnitude >= other.magnitude


def test_drive_audit_v1_accepts_tick_attribution() -> None:
    audit = DriveAuditV1.model_validate(
        {
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.drives.audit.v1",
            "dominant_drive": "predictive",
            "tick_attribution": {"predictive": 0.25, "autonomy": 0.1},
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )
    assert audit.tick_attribution["predictive"] == 0.25
    assert audit.dominant_drive == "predictive"


def test_audit_emits_tick_attribution() -> None:
    """Spec acceptance 6."""
    gap = _gap_tension()
    env = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="test", version="0"),
        correlation_id=uuid4(),
        payload={},
    )
    drive_state = DriveStateV1.model_validate(
        {
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.drives.state.v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pressures": {k: 0.73 for k in DRIVE_KEYS},
            "activations": {k: True for k in DRIVE_KEYS},
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )
    attribution = compute_tick_attribution([gap], metabolism_deltas={"predictive": 0.15})
    dominant = dominant_drive_from_attribution(attribution, lead_tension=gap)
    audit = build_drive_audit(
        env=env,
        intake_channel="orion:world_pulse:run:result",
        drive_state=drive_state,
        tensions=[gap],
        tick_attribution=attribution,
        dominant_drive=dominant,
    )
    assert audit.tick_attribution["predictive"] > 0.0
    assert audit.dominant_drive == "predictive"
