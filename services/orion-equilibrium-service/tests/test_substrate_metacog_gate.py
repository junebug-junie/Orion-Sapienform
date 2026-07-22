from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.schemas.execution_projection import ExecutionRunStateV1, ExecutionTrajectoryProjectionV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from app.substrate_metacog_gate import build_substrate_metacog_trigger


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


def test_build_substrate_metacog_trigger_returns_none_with_production_default_thresholds():
    """2026-07-22 (SelfStateV1 burn): compute_substrate_eventfulness() dropped its
    self_state-derived scoring terms (SelfStateV1 no longer exists), leaving only
    the execution_trajectory term, capped at 0.25. Production's real thresholds
    (dense=0.55, pulse=0.30 -- see settings.metacog_substrate_{dense,pulse}_threshold)
    both sit above that ceiling, so this path cannot fire a trigger at all right
    now, even with real execution failures present. Disclosed here rather than
    hidden behind a passing assertion: a real scoring-term replacement is a
    follow-up, not something this burn attempted.
    """
    execution = _execution_with_failures()

    def _fake_hydrate(ctx):
        ctx["execution_trajectory_projection"] = execution.model_dump(mode="json")

    with patch("app.substrate_metacog_gate.hydrate_felt_state_ctx", side_effect=_fake_hydrate):
        trigger = build_substrate_metacog_trigger(
            zen_state="not_zen",
            pressure=0.4,
            recall_enabled=False,
            dense_threshold=0.55,
            pulse_threshold=0.30,
        )

    assert trigger is None


def test_build_substrate_metacog_trigger_returns_pulse_trigger_with_lowered_threshold():
    execution = _execution_with_failures()

    def _fake_hydrate(ctx):
        ctx["execution_trajectory_projection"] = execution.model_dump(mode="json")

    with patch("app.substrate_metacog_gate.hydrate_felt_state_ctx", side_effect=_fake_hydrate):
        trigger = build_substrate_metacog_trigger(
            zen_state="zen",
            pressure=0.2,
            recall_enabled=False,
            dense_threshold=0.55,
            pulse_threshold=0.20,
        )

    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "pulse"
    assert "execution_failures" in trigger.reason


def test_build_substrate_metacog_trigger_returns_dense_trigger_with_lowered_threshold():
    execution = _execution_with_failures()

    def _fake_hydrate(ctx):
        ctx["execution_trajectory_projection"] = execution.model_dump(mode="json")

    with patch("app.substrate_metacog_gate.hydrate_felt_state_ctx", side_effect=_fake_hydrate):
        trigger = build_substrate_metacog_trigger(
            zen_state="not_zen",
            pressure=0.4,
            recall_enabled=False,
            dense_threshold=0.20,
            pulse_threshold=0.10,
        )

    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "dense"


def test_build_substrate_metacog_trigger_returns_none_when_quiet():
    def _fake_hydrate(ctx):
        pass

    with patch("app.substrate_metacog_gate.hydrate_felt_state_ctx", side_effect=_fake_hydrate):
        trigger = build_substrate_metacog_trigger(
            zen_state="zen",
            pressure=0.1,
            recall_enabled=False,
            dense_threshold=0.55,
            pulse_threshold=0.30,
        )
    assert trigger is None
