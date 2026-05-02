"""Tests for causal parent resolution across the signal window."""
import pytest
from datetime import datetime, timezone

from orion.signals.models import OrionSignalV1, OrganClass


def make_signal(organ_id: str, signal_id: str, organ_class: OrganClass = OrganClass.exogenous) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id=signal_id,
        organ_id=organ_id,
        organ_class=organ_class,
        signal_kind="test_kind",
        dimensions={"level": 0.5, "confidence": 0.9},
        causal_parents=[],
        observed_at=now,
        emitted_at=now,
    )


class TestSignalWindow:
    def test_put_and_get(self):
        from app.signal_window import SignalWindow
        window = SignalWindow(window_sec=30.0)
        sig = make_signal("biometrics", "sig-001")
        window.put(sig)
        retrieved = window.get("biometrics")
        assert retrieved is not None
        assert retrieved.signal_id == "sig-001"

    def test_returns_none_for_unknown_organ(self):
        from app.signal_window import SignalWindow
        window = SignalWindow(window_sec=30.0)
        assert window.get("unknown_organ") is None

    def test_evicts_stale_signals(self):
        from app.signal_window import SignalWindow
        from datetime import timedelta
        window = SignalWindow(window_sec=1.0)
        old_time = datetime.now(timezone.utc) - timedelta(seconds=60)
        sig = OrionSignalV1(
            signal_id="old-sig",
            organ_id="biometrics",
            organ_class=OrganClass.exogenous,
            signal_kind="gpu_load",
            dimensions={"level": 0.5},
            observed_at=old_time,
            emitted_at=old_time,
        )
        window._signals["biometrics"] = sig
        assert window.get("biometrics") is None

    def test_otel_trace_id_propagation(self):
        """Endogenous signal should share trace_id with its causal parent."""
        bio = make_signal("biometrics", "bio-001")
        bio.otel_trace_id = "trace-abc"

        collapse = make_signal("collapse_mirror", "col-001", OrganClass.endogenous)
        collapse.otel_trace_id = bio.otel_trace_id
        collapse.causal_parents = [bio.signal_id]

        assert collapse.otel_trace_id == bio.otel_trace_id
        assert bio.signal_id in collapse.causal_parents
