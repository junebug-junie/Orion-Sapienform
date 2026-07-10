from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from orion.core.schemas.drives import ArtifactProvenance, GoalProposalV1


def _sample_goal_proposal(
    *,
    artifact_id: str = "goal-1",
    correlation_id: str = "c-1",
    priority: float = 0.5,
    proposal_status: str = "proposed",
) -> GoalProposalV1:
    return GoalProposalV1(
        artifact_id=artifact_id,
        subject="orion",
        model_layer="drives",
        entity_id="orion",
        kind="memory.goals.proposed.v1",
        correlation_id=correlation_id,
        provenance=ArtifactProvenance(intake_channel="orion:memory:drives:state"),
        goal_statement="investigate the anomaly",
        proposal_signature="sig-1",
        drive_origin="curiosity",
        priority=priority,
        proposal_status=proposal_status,
    )


@pytest.mark.asyncio
async def test_handle_bus_message_valid_envelope_sets_active_goal(monkeypatch) -> None:
    from app.goal_context_listener import _handle_bus_message
    from app.settings import Settings
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    goal = _sample_goal_proposal(correlation_id="corr-bus")
    corr = str(uuid4())
    envelope = BaseEnvelope(
        kind="memory.goals.proposed.v1",
        source=ServiceRef(name="orion-spark-concept-induction"),
        correlation_id=corr,
        payload=goal.model_copy(update={"correlation_id": corr}).model_dump(mode="json"),
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=envelope, error=None)

    seen: list[GoalProposalV1] = []
    monkeypatch.setattr(
        "orion.substrate.attention.goal_context.set_active_goal",
        lambda g: seen.append(g),
    )

    await _handle_bus_message(
        bus,
        {"data": b"ignored"},
        settings=Settings(POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion"),
    )

    assert len(seen) == 1
    assert seen[0].correlation_id == corr
    assert seen[0].drive_origin == "curiosity"


@pytest.mark.asyncio
async def test_handle_bus_message_decode_failure_is_noop(monkeypatch) -> None:
    from app.goal_context_listener import _handle_bus_message
    from app.settings import Settings

    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=False, envelope=None, error="bad payload")

    called = []
    monkeypatch.setattr(
        "orion.substrate.attention.goal_context.set_active_goal",
        lambda g: called.append(g),
    )

    await _handle_bus_message(
        bus,
        {"data": b"ignored"},
        settings=Settings(POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion"),
    )

    assert called == []


@pytest.mark.asyncio
async def test_handle_bus_message_wrong_kind_is_noop(monkeypatch, caplog) -> None:
    import logging

    from app.goal_context_listener import _handle_bus_message
    from app.settings import Settings
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    goal = _sample_goal_proposal()
    envelope = BaseEnvelope(
        kind="some.other.kind.v1",
        source=ServiceRef(name="orion-spark-concept-induction"),
        correlation_id=str(uuid4()),
        payload=goal.model_dump(mode="json"),
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=envelope, error=None)

    called = []
    monkeypatch.setattr(
        "orion.substrate.attention.goal_context.set_active_goal",
        lambda g: called.append(g),
    )

    caplog.set_level(logging.WARNING, logger="orion.substrate.runtime.goal_context_listener")
    await _handle_bus_message(
        bus,
        {"data": b"ignored"},
        settings=Settings(POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion"),
    )

    assert called == []
    assert "unsupported kind" in caplog.text


@pytest.mark.asyncio
async def test_handle_bus_message_malformed_payload_is_noop(monkeypatch) -> None:
    from app.goal_context_listener import _handle_bus_message
    from app.settings import Settings
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    envelope = BaseEnvelope(
        kind="memory.goals.proposed.v1",
        source=ServiceRef(name="orion-spark-concept-induction"),
        correlation_id=str(uuid4()),
        payload={"not": "a valid goal proposal"},
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=envelope, error=None)

    called = []
    monkeypatch.setattr(
        "orion.substrate.attention.goal_context.set_active_goal",
        lambda g: called.append(g),
    )

    await _handle_bus_message(
        bus,
        {"data": b"ignored"},
        settings=Settings(POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion"),
    )

    assert called == []


@pytest.mark.asyncio
async def test_run_goal_context_listener_returns_when_bus_disabled() -> None:
    from app.goal_context_listener import run_goal_context_listener
    from app.settings import Settings

    settings = Settings(
        POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion",
        ORION_BUS_ENABLED=False,
    )
    bus = MagicMock()

    await run_goal_context_listener(bus, settings=settings)

    bus.subscribe.assert_not_called()
