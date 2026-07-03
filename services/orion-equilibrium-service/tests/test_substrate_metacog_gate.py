from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.schemas.self_state import SelfStateV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from app.substrate_metacog_gate import build_substrate_metacog_trigger


def _dense_self_state() -> SelfStateV1:
    now = datetime.now(timezone.utc)
    return SelfStateV1(
        self_state_id="ss-dense",
        generated_at=now,
        source_field_tick_id="tick-1",
        source_field_generated_at=now,
        source_attention_frame_id="frame-1",
        source_attention_generated_at=now,
        overall_condition="strained",
        overall_intensity=0.7,
        overall_confidence=0.8,
        overall_surprise=0.62,
        trajectory_condition="stable",
        prediction_error_scores={},
    )


def test_build_substrate_metacog_trigger_returns_dense_trigger():
    def _fake_hydrate(ctx):
        ctx["self_state"] = _dense_self_state().model_dump(mode="json")

    with patch("app.substrate_metacog_gate.hydrate_felt_state_ctx", side_effect=_fake_hydrate):
        trigger = build_substrate_metacog_trigger(
            zen_state="not_zen",
            pressure=0.4,
            recall_enabled=False,
            dense_threshold=0.55,
            pulse_threshold=0.30,
        )

    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "dense"
    assert "substrate_eventfulness" in trigger.reason


def test_build_substrate_metacog_trigger_returns_pulse_trigger():
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    from orion.schemas.execution_projection import ExecutionRunStateV1, ExecutionTrajectoryProjectionV1

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

    def _fake_hydrate(ctx):
        ctx["self_state"] = _dense_self_state().model_copy(
            update={
                "overall_surprise": 0.1,
                "overall_condition": "steady",
                "trajectory_condition": "degrading",
            }
        ).model_dump(mode="json")
        ctx["execution_trajectory_projection"] = execution.model_dump(mode="json")

    with patch("app.substrate_metacog_gate.hydrate_felt_state_ctx", side_effect=_fake_hydrate):
        trigger = build_substrate_metacog_trigger(
            zen_state="zen",
            pressure=0.2,
            recall_enabled=False,
            dense_threshold=0.55,
            pulse_threshold=0.30,
        )

    assert isinstance(trigger, MetacogTriggerV1)
    assert trigger.trigger_kind == "pulse"


def test_build_substrate_metacog_trigger_returns_none_when_quiet():
    def _fake_hydrate(ctx):
        ctx["self_state"] = _dense_self_state().model_copy(
            update={"overall_surprise": 0.1, "overall_condition": "steady"}
        ).model_dump(mode="json")

    with patch("app.substrate_metacog_gate.hydrate_felt_state_ctx", side_effect=_fake_hydrate):
        trigger = build_substrate_metacog_trigger(
            zen_state="zen",
            pressure=0.1,
            recall_enabled=False,
            dense_threshold=0.55,
            pulse_threshold=0.30,
        )
    assert trigger is None
