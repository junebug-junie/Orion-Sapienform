"""Wiring: endogenous origination through the real engine, goals, and worker.

Covers spec tests 5 (engine pass-through), 6 (goals drive_origin), 8 (back-compat)
plus the worker flag on/off behavior.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.drives import DriveStateV1, TensionEventV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.spark.concept_induction.drives import DRIVE_KEYS, DriveEngine, DriveMathConfig
from orion.spark.concept_induction.goals import GoalProposalEngine
from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.settings import ConceptSettings

NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def _endo_tension(drive: str = "coherence", mag: float = 0.5) -> TensionEventV1:
    return TensionEventV1(
        subject="orion", model_layer="self-model", entity_id="self:orion",
        kind="tension.endogenous.v1", magnitude=mag, drive_impacts={drive: 1.0},
        origin="endogenous", origination_signal={"drift": 0.9, "P": 0.6},
        provenance={"intake_channel": "substrate.self_state.v1"},
    )


def test_endogenous_tension_moves_drive_via_real_engine() -> None:
    """Spec test 5: a fired endogenous tension traverses the unchanged leaky
    DriveEngine and moves the mapped drive from rest to exactly min(cap, P)."""
    engine = DriveEngine(DriveMathConfig(leaky_math_enabled=True))
    pressures, activations = engine.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_endo_tension("coherence", 0.5)],
        now=NOW, previous_ts=None,
    )
    assert abs(pressures["coherence"] - 0.5) < 1e-9  # single firing = cap, not active
    assert activations["coherence"] is False          # 0.5 < 0.62 activate


def test_goals_drive_origin_endogenous() -> None:
    """Spec test 6: an endogenous lead tension yields drive_origin='endogenous'."""
    ds = DriveStateV1(
        subject="orion", model_layer="self-model", entity_id="self:orion",
        kind="memory.drives.state.v1", pressures={"coherence": 0.5},
        activations={"coherence": False}, provenance={"intake_channel": "x"},
    )
    origin = GoalProposalEngine._drive_origin(
        ds, dominant_drive="coherence", source="tick_attribution", lead_origin="endogenous",
    )
    assert origin == "endogenous"
    # exogenous lead keeps the drive-name origin
    origin2 = GoalProposalEngine._drive_origin(
        ds, dominant_drive="coherence", source="tick_attribution", lead_origin="exogenous",
    )
    assert origin2 == "coherence"


def test_back_compat_tension_without_origin() -> None:
    """Spec test 8: a serialized tension without origin deserializes as exogenous."""
    payload = {
        "artifact_id": "t1", "subject": "orion", "model_layer": "self-model",
        "entity_id": "self:orion", "kind": "substrate.world_coverage_gap",
        "magnitude": 0.5, "drive_impacts": {"predictive": 0.15},
        "provenance": {"intake_channel": "x"},
    }
    t = TensionEventV1.model_validate(payload)
    assert t.origin == "exogenous"
    assert t.origination_signal == {}


def _self_state(*, trajectory=None, agency=0.9, intensity=0.0, dwell=40, unresolved=None) -> SelfStateV1:
    dims = {"agency_readiness": SelfStateDimensionV1(
        dimension_id="agency_readiness", score=agency, confidence=1.0)}
    return SelfStateV1(
        self_state_id=str(uuid4()), generated_at=NOW,
        source_field_tick_id="ft", source_field_generated_at=NOW,
        source_attention_frame_id="af", source_attention_generated_at=NOW,
        overall_intensity=intensity, overall_confidence=0.7, dimensions=dims,
        dimension_trajectory=trajectory or {"coherence": 0.9, "uncertainty": 0.8},
        attention_dwell_ticks=dwell, unresolved_pressures=unresolved or [],
    )


def _self_state_env() -> BaseEnvelope:
    return BaseEnvelope(
        id=uuid4(), kind="substrate.self_state.v1", correlation_id=uuid4(),
        created_at=NOW, source=ServiceRef(name="orion-substrate-runtime", version="0.1.0", node="athena"),
        payload=_self_state().model_dump(mode="json"),
    )


def _worker(enabled: bool, monkeypatch) -> ConceptWorker:
    monkeypatch.setenv("ORION_ENDOGENOUS_ORIGINATION_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("ORIGINATION_COOLDOWN_SEC", "0")  # allow repeated fires in test
    worker = ConceptWorker(ConceptSettings())
    return worker


def test_worker_flag_off_no_origination(monkeypatch) -> None:
    worker = _worker(False, monkeypatch)
    # A quiet self-state (no prior => extract_tensions_from_self_state yields no deltas).
    out = worker._tensions_from_self_state(_self_state_env(), "orion:substrate:self_state")
    assert all(t.origin == "exogenous" for t in out)


def test_worker_flag_on_quiet_fires(monkeypatch) -> None:
    worker = _worker(True, monkeypatch)
    ch = "orion:substrate:self_state"
    # Feed several quiet high-drift self-states to warm the ring, then it should fire.
    fired = False
    for _ in range(8):
        out = worker._tensions_from_self_state(_self_state_env(), ch)
        if any(t.origin == "endogenous" for t in out):
            fired = True
            endo = next(t for t in out if t.origin == "endogenous")
            assert endo.magnitude <= 0.5
            assert endo.drive_impacts  # maps a drive
            break
    assert fired, "endogenous origination should fire on a quiet high-drift stream"
