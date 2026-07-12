from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

import app.worker as worker
from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy

_NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def setup_function(_fn) -> None:
    worker._LATEST_SELF_STATE = None


def teardown_function(_fn) -> None:
    worker._LATEST_SELF_STATE = None


def _dim(name: str, score: float, *, dominant_evidence: list[str] | None = None) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(
        dimension_id=name,
        score=score,
        confidence=1.0,
        dominant_evidence=dominant_evidence or [],
    )


def _self_state_payload(
    *,
    resource_dim: SelfStateDimensionV1,
    execution_dim: SelfStateDimensionV1 | None = None,
    dimension_trajectory: dict[str, float] | None = None,
) -> dict:
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
            ("transport_integrity", 1.0),
        )
    }
    dims["resource_pressure"] = resource_dim
    if execution_dim is not None:
        dims["execution_pressure"] = execution_dim
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
        dimension_trajectory=dimension_trajectory or {},
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


def test_hardware_resource_pressure_ignores_non_finite_values() -> None:
    """dominant_evidence is an unvalidated list[str] crossing a service
    boundary. A NaN/inf entry must be dropped, not silently pass float()
    and later poison resource_cap ** 0.5 into a complex number."""
    dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["cpu_pressure=nan", "gpu_pressure=inf", "memory_pressure=0.30"],
    )
    assert worker._hardware_resource_pressure(dim) == 0.30


def test_hardware_resource_pressure_clamps_out_of_range_values() -> None:
    """A malformed >1.0 hardware value must be clamped, not passed through
    raw -- an unclamped value would drive resource_cap negative and
    (resource_cap * execution_cap) ** 0.5 into complex-number territory."""
    dim = _dim("resource_pressure", 1.0, dominant_evidence=["cpu_pressure=1.50"])
    assert worker._hardware_resource_pressure(dim) == 1.0


def test_energy_momentum_ignores_raw_resource_pressure_trajectory_when_hardware_evidence_used() -> None:
    """Regression: dimension_trajectory["resource_pressure"] tracks the delta
    of the raw, still-poisoned aggregate score -- the same untraced channel
    the base term was fixed to bypass. When hardware evidence drives the base
    term, the raw trajectory delta must NOT also feed the momentum term, or
    the stuck channel's theater re-enters through the back door."""
    resource_dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["pressure=1.00", "cpu_pressure=0.92"],
    )
    payload = _self_state_payload(
        resource_dim=resource_dim,
        dimension_trajectory={"resource_pressure": 1.0, "field_intensity": 0.0},
    )
    ss = SelfStateV1.model_validate(payload)
    phi = worker._phi_from_self_state(ss)
    # Same 0.2828 as the no-trajectory case: the large resource_pressure
    # trajectory delta must be ignored while hardware evidence is in use.
    assert phi["energy"] == 0.2828


def test_energy_momentum_still_uses_raw_trajectory_in_fallback_path() -> None:
    """When there's no hardware evidence (fallback to the raw score), the
    trajectory momentum term still applies as before this fix -- only the
    hardware-evidence path skips it."""
    resource_dim = _dim("resource_pressure", 0.5, dominant_evidence=[])
    payload = _self_state_payload(
        resource_dim=resource_dim,
        dimension_trajectory={"resource_pressure": 0.2, "field_intensity": 0.0},
    )
    ss = SelfStateV1.model_validate(payload)
    phi = worker._phi_from_self_state(ss)
    # intensity=1.0, resource_cap=1-0.5=0.5, execution_cap=1.0
    # base = 1.0 * (0.5 * 1.0) ** 0.5 ~= 0.7071
    # momentum = -0.1 * max(0, 0.2) = -0.02 -> 0.6871
    assert phi["energy"] == 0.6871


@pytest.mark.asyncio
async def test_handle_semantic_upsert_arousal_matches_canonical_energy(monkeypatch) -> None:
    """Regression: handle_semantic_upsert's tissue.update broadcast used to
    compute its own bespoke arousal proxy (0.15 + 1.2*novelty) instead of the
    canonical phi_stats["energy"] every other broadcast site uses
    (handle_self_state, handle_trace) -- inconsistent within the same
    process. Confirms the broadcast now reports the same energy value
    _phi_from_self_state() computes."""
    monkeypatch.setattr(worker, "_EXPECTED_EMB", {}, raising=False)
    monkeypatch.setattr(worker, "_SEEN_DOC", {}, raising=False)
    monkeypatch.setattr(worker, "_pub_bus", None, raising=False)
    monkeypatch.setattr(worker, "_ACTIVE_SIGNALS", [], raising=False)

    resource_dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["pressure=1.00", "cpu_pressure=0.92", "memory_pressure=0.05"],
    )
    ss = SelfStateV1.model_validate(_self_state_payload(resource_dim=resource_dim))
    worker.set_latest_self_state(ss)

    broadcasts = []

    async def _capture_broadcast(payload):
        broadcasts.append(payload)

    monkeypatch.setattr(worker.manager, "broadcast", _capture_broadcast, raising=False)

    env = BaseEnvelope(
        kind="vector.upsert.v1",
        source=ServiceRef(name="orion-vector-host", node="n1"),
        correlation_id=uuid4(),
        payload={
            "doc_id": "turn-arousal-1",
            "collection": "orion_chat_turns",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "embedding_kind": "semantic",
            "text": "hello world",
            "meta": {},
        },
    )

    await worker.handle_semantic_upsert(env)

    tissue = [b for b in broadcasts if b.get("type") == "tissue.update"]
    assert tissue, "expected a tissue.update broadcast"
    # Same 0.2828 as _phi_from_self_state's direct test: intensity=1.0,
    # resource_cap=1-0.92=0.08, execution_cap=1.0, no trajectory deltas,
    # no active signals.
    assert tissue[-1]["stats"]["arousal"] == pytest.approx(0.2828, abs=1e-4)


def test_execution_load_pressure_none_when_only_generic_channel_present() -> None:
    dim = _dim("execution_pressure", 1.0, dominant_evidence=["execution_pressure=1.00"])
    assert worker._execution_load_pressure(dim) is None


def test_execution_load_pressure_prefers_execution_load_over_saturated_generic() -> None:
    dim = _dim(
        "execution_pressure",
        1.0,
        dominant_evidence=["execution_pressure=1.00", "execution_load=0.19"],
    )
    assert worker._execution_load_pressure(dim) == 0.19


def test_arousal_no_longer_hard_zeroed_by_saturated_execution_pressure_channel() -> None:
    """Live 2026-07-10 follow-on: after resource_pressure hardware filter,
    energy was still hard-zeroed because execution_pressure.score=1.0 from a
    stuck generic `execution_pressure` channel while real execution_load≈0.19.
    With execution_load in dominant_evidence, energy must reflect real load."""
    resource_dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["pressure=1.00", "cpu_pressure=0.92"],
    )
    execution_dim = _dim(
        "execution_pressure",
        1.0,
        dominant_evidence=["execution_pressure=1.00", "execution_load=0.19"],
    )
    ss = SelfStateV1.model_validate(
        _self_state_payload(resource_dim=resource_dim, execution_dim=execution_dim)
    )
    phi = worker._phi_from_self_state(ss)
    assert phi["energy"] > 0.0
    # intensity=1.0, resource_cap=1-0.92=0.08, execution_cap=1-0.19=0.81
    # energy = 1.0 * (0.08 * 0.81) ** 0.5 ~= 0.2546
    assert phi["energy"] == pytest.approx(0.2546, abs=1e-4)


def test_arousal_still_zero_when_no_execution_load_evidence_and_score_saturated() -> None:
    """Honest fallback: no execution_load in evidence → use raw dim.score."""
    resource_dim = _dim(
        "resource_pressure",
        1.0,
        dominant_evidence=["cpu_pressure=0.50"],
    )
    execution_dim = _dim(
        "execution_pressure", 1.0, dominant_evidence=["execution_pressure=1.00"]
    )
    ss = SelfStateV1.model_validate(
        _self_state_payload(resource_dim=resource_dim, execution_dim=execution_dim)
    )
    phi = worker._phi_from_self_state(ss)
    assert phi["energy"] == 0.0


def test_hardware_bypass_survives_real_build_self_state_pipeline() -> None:
    """End-to-end regression guard (2026-07-12, found by code review): the
    tests above hand-construct dominant_evidence directly, which doesn't
    catch a break in the real producer. This test runs the actual
    orion-self-state-runtime pipeline (build_self_state) and confirms its
    output still lets _hardware_resource_pressure/_execution_load_pressure
    find real hardware evidence -- proving evidence_channel_map actually
    closes the loop, not just that a hand-written fixture says so."""
    repo_root = Path(__file__).resolve().parents[3]
    attention_policy = load_attention_policy(
        repo_root / "config" / "attention" / "field_attention_policy.v1.yaml"
    )
    self_state_policy = load_self_state_policy(
        repo_root / "config" / "self_state" / "self_state_policy.v1.yaml"
    )
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    field = FieldStateV1(
        generated_at=now,
        tick_id="tick_e2e_hardware_bypass",
        node_vectors={"node:circe": {"gpu_pressure": 1.0}},
        capability_vectors={"capability:llm_inference": {"pressure": 0.50}},
    )
    attention = build_attention_frame(field=field, policy=attention_policy, now=now)
    ss = build_self_state(field=field, attention=attention, policy=self_state_policy, now=now)

    hardware_pressure = worker._hardware_resource_pressure(ss.dimensions.get("resource_pressure"))
    assert hardware_pressure == 1.0
    phi = worker._phi_from_self_state(ss)
    # resource_cap = 1 - 1.0 (real hardware pressure) = 0.0, so energy is
    # capped by the real (saturated) hardware signal, not silently falling
    # back to a possibly-different raw dimension score.
    assert phi["energy"] == 0.0
