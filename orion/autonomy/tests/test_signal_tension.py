"""Task 4: signal/failure/health -> deviation-driven tensions."""
from __future__ import annotations

from datetime import datetime, timezone

from orion.autonomy.deviation_gate import DeviationGate
from orion.autonomy.models import AutonomyEvidenceRefV1
from orion.autonomy.signal_drive_map import load_signal_drive_map
from orion.autonomy.signal_tension import (
    chat_evidence_to_tension,
    equilibrium_to_tension,
    failure_to_tension,
    signal_to_tension,
)
from orion.signals.models import OrganClass, OrionSignalV1

SDM = load_signal_drive_map()


def _sig(kind: str, dims: dict, *, sid: str = "s1", notes=None, source="src-1") -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id=sid,
        organ_id="test-organ",
        organ_class=OrganClass.exogenous,
        signal_kind=kind,
        dimensions=dims,
        source_event_id=source,
        observed_at=now,
        emitted_at=now,
        notes=notes or [],
    )


def _gate() -> DeviationGate:
    return DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02, impulse_k=0.25, warmup=5)


def test_stub_signal_dropped() -> None:
    stub = _sig("biometrics_state", {"level": 0.5, "confidence": 0.5}, source=None)
    assert signal_to_tension(stub, _gate(), SDM) is None


def test_steady_signal_mints_nothing() -> None:
    gate = _gate()
    out = None
    for _ in range(50):
        out = signal_to_tension(_sig("spark_signal", {"coherence": 0.8, "confidence": 0.9}), gate, SDM)
    assert out is None


def test_real_coherence_drop_mints_coherence_tension() -> None:
    gate = _gate()
    for _ in range(30):
        signal_to_tension(_sig("spark_signal", {"coherence": 0.85, "confidence": 0.9}), gate, SDM)
    t = signal_to_tension(_sig("spark_signal", {"coherence": 0.45, "confidence": 0.9}), gate, SDM)
    assert t is not None
    assert t.kind == "tension.signal.v1"
    assert t.drive_impacts.get("coherence", 0.0) > 0.0
    assert t.magnitude > 0.0


def test_scene_state_flood_unmapped_mints_nothing() -> None:
    gate = _gate()
    outs = [signal_to_tension(_sig("scene_state", {"salience": 0.9, "confidence": 0.9}), gate, SDM)
            for _ in range(100)]
    assert all(o is None for o in outs)


def test_biometric_metric_drop_maps_via_suffix() -> None:
    gate = _gate()
    for _ in range(30):
        signal_to_tension(_sig("biometrics_state", {"hrv_level": 0.8, "confidence": 0.9}), gate, SDM)
    t = signal_to_tension(_sig("biometrics_state", {"hrv_level": 0.4, "confidence": 0.9}), gate, SDM)
    assert t is not None
    assert t.drive_impacts.get("capability", 0.0) > 0.0


def test_failure_maps_directly() -> None:
    t = failure_to_tension(severity=0.9, sdm=SDM, channel="orion:system:error", summary="boom")
    assert t is not None
    assert t.kind == "tension.failure.v1"
    assert t.drive_impacts.get("capability", 0.0) > 0.0
    # Zero severity mints nothing.
    assert failure_to_tension(severity=0.0, sdm=SDM, channel="x") is None


def test_equilibrium_edge_triggered() -> None:
    # ok -> degraded mints once.
    t1 = equilibrium_to_tension(healthy=False, prev_healthy=True, sdm=SDM)
    assert t1 is not None and t1.kind == "tension.health.v1"
    # degraded -> degraded mints nothing.
    assert equilibrium_to_tension(healthy=False, prev_healthy=False, sdm=SDM) is None
    # recovered mints nothing.
    assert equilibrium_to_tension(healthy=True, prev_healthy=False, sdm=SDM) is None


def test_never_raises_on_garbage() -> None:
    assert signal_to_tension(_sig("spark_signal", {}), _gate(), SDM) is None


def test_chat_evidence_mapped_hazard_mints_tension() -> None:
    ev = AutonomyEvidenceRefV1(
        evidence_id="h1",
        source="social_bridge",
        kind="relational_signal",
        summary="cooldown_active",
        confidence=0.6,
        observed_at=datetime(2026, 7, 10, 12, 0, 0),
        signal_kind="chat_social_hazard",
        dimension="cooldown_active",
        value=1.0,
    )
    t = chat_evidence_to_tension(ev, SDM)
    assert t is not None
    assert t.kind == "tension.chat_evidence.v1"
    assert t.magnitude > 0.0
    assert t.drive_impacts.get("relational", 0.0) > 0.0


def test_chat_evidence_unmapped_or_missing_fields_returns_none() -> None:
    bare = AutonomyEvidenceRefV1(
        evidence_id="h2",
        source="social_bridge",
        kind="relational_signal",
        summary="context_excluded:memory",
        confidence=0.6,
    )
    assert chat_evidence_to_tension(bare, SDM) is None

    unmapped = AutonomyEvidenceRefV1(
        evidence_id="h3",
        source="social_bridge",
        kind="relational_signal",
        summary="peer_targeted_elsewhere",
        signal_kind="chat_social_hazard",
        dimension="peer_targeted_elsewhere",
        value=1.0,
    )
    assert chat_evidence_to_tension(unmapped, SDM) is None


def test_chat_evidence_zero_value_returns_none() -> None:
    ev = AutonomyEvidenceRefV1(
        evidence_id="h4",
        source="reasoning",
        kind="reasoning_quality",
        summary="fallback",
        signal_kind="chat_reasoning_quality",
        dimension="fallback",
        value=0.0,
    )
    assert chat_evidence_to_tension(ev, SDM) is None


def test_chat_evidence_never_raises_on_garbage() -> None:
    class Boom:
        signal_kind = "chat_social_hazard"
        dimension = "cooldown_active"
        evidence_id = "x"
        summary = "x"

        @property
        def value(self):
            raise RuntimeError("boom")

    assert chat_evidence_to_tension(Boom(), SDM) is None  # type: ignore[arg-type]
