from __future__ import annotations

from datetime import datetime, timezone

import app.worker as worker
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

_NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def _dim(name: str, score: float, *, dominant_evidence: list[str] | None = None) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(
        dimension_id=name,
        score=score,
        confidence=1.0,
        dominant_evidence=dominant_evidence or [],
    )


def _self_state_payload(*, resource_dim: SelfStateDimensionV1) -> dict:
    dims = {
        name: _dim(name, score)
        for name, score in (
            ("coherence", 1.0),
            ("field_intensity", 1.0),
            ("agency_readiness", 0.5),
            ("execution_pressure", 0.0),
            ("reasoning_pressure", 0.1),
            ("reliability_pressure", 0.0),
            ("continuity_pressure", 0.0),
            ("social_pressure", 0.0),
            ("introspection_pressure", 0.0),
            ("uncertainty", 0.0),
            ("policy_pressure", 0.0),
            ("transport_integrity", 1.0),
        )
    }
    dims["resource_pressure"] = resource_dim
    return SelfStateV1(
        self_state_id="self.state:tick_arousal_test:policy.v1",
        generated_at=_NOW,
        source_field_tick_id="tick_arousal_test",
        source_field_generated_at=_NOW,
        source_attention_frame_id="frame_arousal_test",
        source_attention_generated_at=_NOW,
        overall_intensity=0.5,
        overall_confidence=0.8,
        overall_condition="steady",
        dimensions=dims,
        dimension_trajectory={},
    ).model_dump(mode="json")


def test_hardware_resource_pressure_none_when_dim_missing() -> None:
    assert worker._hardware_resource_pressure(None) is None


def test_hardware_resource_pressure_none_when_only_generic_channel_present() -> None:
    dim = _dim("resource_pressure", 1.0, dominant_evidence=["pressure=1.00"])
    assert worker._hardware_resource_pressure(dim) is None


def test_hardware_resource_pressure_ignores_generic_and_transport_channels() -> None:
    dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["pressure=1.00", "cpu_pressure=0.92", "transport_pressure=0.10"],
    )
    assert worker._hardware_resource_pressure(dim) == 0.92


def test_hardware_resource_pressure_takes_max_of_multiple_hardware_channels() -> None:
    dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["gpu_pressure=0.40", "cpu_pressure=0.92", "memory_pressure=0.10"],
    )
    assert worker._hardware_resource_pressure(dim) == 0.92


def test_arousal_no_longer_hard_zeroed_by_saturated_generic_pressure_channel() -> None:
    """Regression for the live incident: resource_pressure.score pinned at 1.0
    by the untraced capability-graph `pressure` channel used to hard-zero
    tissue-viz energy/arousal regardless of real hardware load. With real
    cpu_pressure=0.92 present in dominant_evidence, energy must reflect the
    genuine 8% headroom instead of 0."""
    resource_dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["pressure=1.00", "cpu_pressure=0.92", "memory_pressure=0.05"],
    )
    ss = SelfStateV1.model_validate(_self_state_payload(resource_dim=resource_dim))
    phi = worker._phi_from_self_state(ss)
    assert phi["energy"] > 0.0
    # intensity=1.0, resource_cap=1-0.92=0.08, execution_cap=1-0.0=1.0
    # energy = 1.0 * (0.08 * 1.0) ** 0.5 ~= 0.2828 (trajectory terms are 0
    # here since dimension_trajectory is empty).
    assert phi["energy"] == 0.2828


def test_arousal_still_zero_when_no_hardware_evidence_present() -> None:
    """Honest no-signal fallback: when resource_pressure carries no hardware
    channel in its evidence (e.g. a leaner payload), behavior is unchanged
    from before this fix -- uses the raw (possibly saturated) score rather
    than fabricating a value."""
    resource_dim = _dim("resource_pressure", 1.0, dominant_evidence=[])
    ss = SelfStateV1.model_validate(_self_state_payload(resource_dim=resource_dim))
    phi = worker._phi_from_self_state(ss)
    assert phi["energy"] == 0.0
