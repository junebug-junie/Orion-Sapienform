"""Task 4: signal/failure/health -> deviation-driven tensions."""
from __future__ import annotations

from datetime import datetime, timezone

from orion.autonomy.deviation_gate import DeviationGate
from orion.autonomy.signal_drive_map import load_signal_drive_map
from orion.autonomy.signal_tension import (
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
