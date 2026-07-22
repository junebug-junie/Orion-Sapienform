from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionRunStateV1, ExecutionTrajectoryProjectionV1
from orion.substrate.metacog_trigger_signals import (
    build_metacog_substrate_cue,
    compute_substrate_eventfulness,
)


def _execution_with_failures() -> ExecutionTrajectoryProjectionV1:
    now = datetime.now(timezone.utc)
    return ExecutionTrajectoryProjectionV1(
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


def test_compute_substrate_eventfulness_pulse_on_execution_failures() -> None:
    """2026-07-22 (SelfStateV1 burn): self_state-derived scoring removed --
    execution_failures alone (0.25) is now the only real contribution, which
    clears the pulse threshold (0.30's default is actually above 0.25, so
    this test uses a lowered pulse_threshold matching the real remaining
    signal's ceiling)."""
    ev = compute_substrate_eventfulness(
        execution_trajectory=_execution_with_failures(), pulse_threshold=0.20
    )
    assert ev.trigger_kind == "pulse"
    assert ev.score >= 0.20
    assert "execution_failures" in ev.reasons


def test_compute_substrate_eventfulness_quiet_returns_none_kind() -> None:
    ev = compute_substrate_eventfulness()
    assert ev.trigger_kind is None
    assert ev.score == 0.0
    assert ev.reasons == ()


def test_build_metacog_substrate_cue_compact() -> None:
    ctx = {"execution_trajectory_projection": _execution_with_failures().model_dump(mode="json")}
    cue = build_metacog_substrate_cue(ctx)
    assert "execution: failed_runs=1" in cue
    assert len(cue) <= 400


def test_build_metacog_substrate_cue_empty_when_no_signal() -> None:
    assert build_metacog_substrate_cue({}) == ""


def test_build_metacog_substrate_cue_tags_its_own_provenance() -> None:
    ctx = {"execution_trajectory_projection": _execution_with_failures().model_dump(mode="json")}
    cue = build_metacog_substrate_cue(ctx)
    assert "source=live_runtime_projection" in cue
