from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionRunStateV1, ExecutionTrajectoryProjectionV1
from orion.schemas.self_state import SelfStateV1
from orion.substrate.metacog_trigger_signals import (
    build_metacog_substrate_cue,
    compute_substrate_eventfulness,
)


def _self_state(**overrides) -> SelfStateV1:
    now = datetime.now(timezone.utc)
    base = dict(
        self_state_id="ss-test",
        generated_at=now,
        source_field_tick_id="tick-1",
        source_field_generated_at=now,
        source_attention_frame_id="frame-1",
        source_attention_generated_at=now,
        overall_condition="steady",
        overall_intensity=0.4,
        overall_confidence=0.8,
        trajectory_condition="stable",
        prediction_error_scores={},
        overall_surprise=0.1,
    )
    base.update(overrides)
    return SelfStateV1(**base)


def test_compute_substrate_eventfulness_dense_on_surprise_and_strain():
    ss = _self_state(overall_surprise=0.6, overall_condition="strained")
    ev = compute_substrate_eventfulness(self_state=ss)
    assert ev.trigger_kind == "dense"
    assert ev.score >= 0.55
    assert any("overall_surprise" in r for r in ev.reasons)


def test_compute_substrate_eventfulness_pulse_on_execution_failures():
    now = datetime.now(timezone.utc)
    execution = ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory",
        generated_at=now,
        runs={
            "run-a": ExecutionRunStateV1(
                trace_id="trace-1",
                correlation_id="corr-1",
                node_id="node-1",
                failed_step_count=2,
                last_updated_at=now,
            )
        },
    )
    ss = _self_state(trajectory_condition="degrading")
    ev = compute_substrate_eventfulness(self_state=ss, execution_trajectory=execution)
    assert ev.trigger_kind == "pulse"
    assert ev.score >= 0.30
    assert "execution_failures" in ev.reasons


def test_compute_substrate_eventfulness_quiet_returns_none_kind():
    ss = _self_state()
    ev = compute_substrate_eventfulness(self_state=ss)
    assert ev.trigger_kind is None
    assert ev.score < 0.30


def test_build_metacog_substrate_cue_compact():
    ss = _self_state(
        overall_condition="strained",
        overall_surprise=0.62,
        trajectory_condition="degrading",
    )
    ctx = {"self_state": ss.model_dump(mode="json")}
    cue = build_metacog_substrate_cue(ctx)
    assert "self_state:" in cue
    assert "strained" in cue
    assert len(cue) <= 400


def test_build_metacog_substrate_cue_tags_its_own_provenance():
    ss = _self_state()
    ctx = {"self_state": ss.model_dump(mode="json")}
    cue = build_metacog_substrate_cue(ctx)
    assert "source=live_runtime_projection" in cue
