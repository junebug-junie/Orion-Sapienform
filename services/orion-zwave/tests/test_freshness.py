from __future__ import annotations

from orion.schemas.telemetry.home_cooling import CoolingObservedStateV1


def test_state_carries_explicit_stale_flag():
    assert CoolingObservedStateV1().stale is None
    assert CoolingObservedStateV1(stale=True).stale is True
