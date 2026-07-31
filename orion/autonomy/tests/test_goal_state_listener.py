from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from orion.autonomy import goal_state
from orion.autonomy.goal_state_listener import _handle_bus_message
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.field_goal import FieldGoalProvenanceV1


def _goal() -> FieldGoalProvenanceV1:
    return FieldGoalProvenanceV1(
        artifact_id="goal-1",
        subject="attention",
        model_layer="field_attention",
        entity_id="node:substrate.biometrics",
        kind="memory.field_goals.proposed.v1",
        field_target_id="node:substrate.biometrics",
        target_kind="node",
        salience_score=0.8,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        priority=0.8,
        proposal_status="proposed",
        provenance={"intake_channel": "internal.attention_runtime"},
    )


@pytest.mark.asyncio
async def test_handle_bus_message_valid_envelope_updates_goal_state(monkeypatch) -> None:
    goal_state.clear_active_goal()
    goal = _goal()
    corr = str(uuid4())
    envelope = BaseEnvelope(
        kind="memory.field_goals.proposed.v1",
        source=ServiceRef(name="orion-attention-runtime"),
        correlation_id=corr,
        payload=goal.model_dump(mode="json"),
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=envelope, error=None)

    await _handle_bus_message(bus, {"data": b"ignored"})

    current = goal_state.get_active_goal()
    assert current is not None
    assert current.artifact_id == "goal-1"
    goal_state.clear_active_goal()


@pytest.mark.asyncio
async def test_handle_bus_message_decode_failure_is_noop() -> None:
    goal_state.clear_active_goal()
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=False, envelope=None, error="bad payload")

    await _handle_bus_message(bus, {"data": b"ignored"})

    assert goal_state.get_active_goal() is None


@pytest.mark.asyncio
async def test_handle_bus_message_wrong_kind_is_noop() -> None:
    goal_state.clear_active_goal()
    goal = _goal()
    envelope = BaseEnvelope(
        kind="some.other.kind.v1",
        source=ServiceRef(name="orion-attention-runtime"),
        correlation_id=str(uuid4()),
        payload=goal.model_dump(mode="json"),
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=envelope, error=None)

    await _handle_bus_message(bus, {"data": b"ignored"})

    assert goal_state.get_active_goal() is None


@pytest.mark.asyncio
async def test_handle_bus_message_malformed_payload_is_noop() -> None:
    goal_state.clear_active_goal()
    envelope = BaseEnvelope(
        kind="memory.field_goals.proposed.v1",
        source=ServiceRef(name="orion-attention-runtime"),
        correlation_id=str(uuid4()),
        payload={"not": "a valid field goal"},
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=envelope, error=None)

    await _handle_bus_message(bus, {"data": b"ignored"})

    assert goal_state.get_active_goal() is None
