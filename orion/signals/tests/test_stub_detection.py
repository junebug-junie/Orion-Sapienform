from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.stub_detection import is_stub_signal


def test_is_stub_signal_detects_placeholder_dimensions() -> None:
    now = datetime.now(timezone.utc)
    sig = OrionSignalV1(
        signal_id="x",
        organ_id="recall",
        organ_class=OrganClass.endogenous,
        signal_kind="recall_result",
        dimensions={"level": 0.5, "confidence": 0.5},
        observed_at=now,
        emitted_at=now,
        notes=["stub adapter — not yet implemented"],
    )
    assert is_stub_signal(sig) is True


def test_is_stub_signal_false_for_real_cognition_step() -> None:
    now = datetime.now(timezone.utc)
    sig = OrionSignalV1(
        signal_id="s1",
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        signal_kind="cognition_step",
        dimensions={"success": 1.0},
        source_event_id="corr-1",
        observed_at=now,
        emitted_at=now,
    )
    assert is_stub_signal(sig) is False


def test_is_stub_signal_stub_organ_without_source_event_id() -> None:
    now = datetime.now(timezone.utc)
    sig = OrionSignalV1(
        signal_id="eq1",
        organ_id="equilibrium",
        organ_class=OrganClass.hybrid,
        signal_kind="mesh_health",
        dimensions={"level": 0.7, "confidence": 0.9},
        observed_at=now,
        emitted_at=now,
    )
    assert is_stub_signal(sig) is True
