from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.worker import AttentionRuntimeWorker
from orion.attention.field_attention.goal_provenance import DominanceStreak
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_goal import FieldGoalProvenanceV1


def _target(target_id: str, salience: float, kind: str = "node") -> FieldAttentionTargetV1:
    return FieldAttentionTargetV1(
        target_id=target_id,
        target_kind=kind,
        salience_score=salience,
        pressure_score=salience,
        novelty_score=salience,
        urgency_score=salience,
        confidence_score=1.0,
    )


def _frame(node_targets: list[FieldAttentionTargetV1]) -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="frame-1",
        generated_at="2026-07-30T00:00:00Z",
        source_field_tick_id="tick-1",
        source_field_generated_at="2026-07-30T00:00:00Z",
        overall_salience=max((t.salience_score for t in node_targets), default=0.0),
        dominant_targets=node_targets,
        node_targets=node_targets,
    )


def _make_worker(monkeypatch, *, producer_enabled: bool = True, min_streak: int = 3) -> AttentionRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_PRODUCER_ENABLED", str(producer_enabled))
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_MIN_STREAK", str(min_streak))
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._node_streak = DominanceStreak()
    worker._bus = MagicMock()
    return worker


def test_maybe_build_goal_returns_none_when_producer_disabled(monkeypatch):
    worker = _make_worker(monkeypatch, producer_enabled=False)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    assert worker._maybe_build_goal(frame) is None


def test_maybe_build_goal_returns_none_when_bus_absent(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker._bus = None
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    assert worker._maybe_build_goal(frame) is None


def test_maybe_build_goal_returns_none_before_streak_threshold(monkeypatch):
    worker = _make_worker(monkeypatch, min_streak=3)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    assert worker._maybe_build_goal(frame) is None  # streak=1
    assert worker._maybe_build_goal(frame) is None  # streak=2


def test_maybe_build_goal_returns_real_goal_at_streak_threshold(monkeypatch):
    worker = _make_worker(monkeypatch, min_streak=3)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.77)])

    worker._maybe_build_goal(frame)  # streak=1
    worker._maybe_build_goal(frame)  # streak=2
    goal = worker._maybe_build_goal(frame)  # streak=3

    assert isinstance(goal, FieldGoalProvenanceV1)
    assert goal.field_target_id == real_domain
    assert goal.target_kind == "node"
    assert goal.salience_score == pytest.approx(0.77)
    assert goal.priority == pytest.approx(0.77)
    assert goal.source_field_tick_id == "tick-1"
    assert goal.source_attention_frame_id == "frame-1"
    assert goal.proposal_status == "proposed"


def test_maybe_build_goal_ignores_host_only_frame(monkeypatch):
    # Candidate B host target only -- no real node:substrate.* domain present.
    worker = _make_worker(monkeypatch, min_streak=1)
    frame = _frame([_target("node:athena", 0.95)])

    assert worker._maybe_build_goal(frame) is None


@pytest.mark.asyncio
async def test_publish_goal_calls_publish_with_reconnect(monkeypatch):
    worker = _make_worker(monkeypatch)
    goal = FieldGoalProvenanceV1(
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
        provenance={"intake_channel": "internal.attention_runtime"},
    )

    mock_publish = AsyncMock()
    monkeypatch.setattr("orion.core.bus.resilience.publish_with_reconnect", mock_publish)

    await worker._publish_goal(goal)

    mock_publish.assert_called_once()
    args, kwargs = mock_publish.call_args
    assert args[0] is worker._bus
    assert args[1] == "orion:memory:goals:proposed"
    assert kwargs.get("log_label") == "attention_runtime_goal_provenance"


@pytest.mark.asyncio
async def test_publish_goal_noop_when_bus_absent(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker._bus = None
    goal = FieldGoalProvenanceV1(
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
        provenance={"intake_channel": "internal.attention_runtime"},
    )

    mock_publish = AsyncMock()
    monkeypatch.setattr("orion.core.bus.resilience.publish_with_reconnect", mock_publish)

    await worker._publish_goal(goal)

    mock_publish.assert_not_called()


@pytest.mark.asyncio
async def test_start_connects_bus_when_producer_and_bus_enabled(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_PRODUCER_ENABLED", "true")
    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._policy = MagicMock()
    worker._health_monitor = MagicMock()
    worker._node_streak = DominanceStreak()
    worker._bus = None
    import asyncio

    worker._stop = asyncio.Event()

    mock_bus_instance = AsyncMock()
    mock_bus_cls = MagicMock(return_value=mock_bus_instance)
    monkeypatch.setattr("orion.core.bus.async_service.OrionBusAsync", mock_bus_cls)
    monkeypatch.setattr(asyncio, "create_task", MagicMock())

    await worker.start()

    mock_bus_cls.assert_called_once()
    mock_bus_instance.connect.assert_awaited_once()
    assert worker._bus is mock_bus_instance


@pytest.mark.asyncio
async def test_start_skips_bus_when_producer_disabled(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_PRODUCER_ENABLED", "false")
    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._policy = MagicMock()
    worker._health_monitor = MagicMock()
    worker._node_streak = DominanceStreak()
    worker._bus = None
    import asyncio

    worker._stop = asyncio.Event()

    mock_bus_cls = MagicMock()
    monkeypatch.setattr("orion.core.bus.async_service.OrionBusAsync", mock_bus_cls)
    monkeypatch.setattr(asyncio, "create_task", MagicMock())

    await worker.start()

    mock_bus_cls.assert_not_called()
    assert worker._bus is None


@pytest.mark.asyncio
async def test_stop_awaits_poll_task_before_closing_bus(monkeypatch):
    """Regression: stop() must not tear down the bus connection while the poll
    loop could still be mid-publish (a real race a review pass on this same
    patch found -- publish_with_reconnect would silently reconnect right after
    an intentional close). Ordering, not just "both eventually happen."
    """
    worker = _make_worker(monkeypatch)
    events: list[str] = []

    async def fake_poll_task() -> None:
        await asyncio.sleep(0.01)
        events.append("poll_task_done")

    worker._poll_task = asyncio.create_task(fake_poll_task())

    async def fake_close() -> None:
        events.append("bus_closed")

    worker._bus.close = AsyncMock(side_effect=fake_close)
    worker._stop = asyncio.Event()

    await worker.stop()

    assert events == ["poll_task_done", "bus_closed"]
