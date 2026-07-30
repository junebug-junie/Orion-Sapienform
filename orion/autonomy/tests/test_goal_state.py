from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.autonomy.goal_state import ActiveGoalStore
from orion.schemas.field_goal import FieldGoalProvenanceV1


def _goal(*, proposal_status: str = "proposed", artifact_id: str = "goal-1") -> FieldGoalProvenanceV1:
    return FieldGoalProvenanceV1(
        artifact_id=artifact_id,
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
        proposal_status=proposal_status,
        provenance={"intake_channel": "internal.attention_runtime"},
    )


def test_active_goal_store_starts_empty() -> None:
    store = ActiveGoalStore()
    assert store.current() is None


def test_active_goal_store_adopts_active_goal() -> None:
    store = ActiveGoalStore()
    store.update_from_goal(_goal())
    current = store.current()
    assert current is not None
    assert current.artifact_id == "goal-1"


def test_active_goal_store_clears_on_terminal_status_for_held_goal() -> None:
    store = ActiveGoalStore()
    store.update_from_goal(_goal(artifact_id="goal-1"))
    store.update_from_goal(_goal(artifact_id="goal-1", proposal_status="completed"))
    assert store.current() is None


def test_active_goal_store_ignores_terminal_status_for_different_goal() -> None:
    store = ActiveGoalStore()
    store.update_from_goal(_goal(artifact_id="goal-1"))
    store.update_from_goal(_goal(artifact_id="goal-2", proposal_status="completed"))
    current = store.current()
    assert current is not None
    assert current.artifact_id == "goal-1"


def test_active_goal_store_staleness_dead_mans_switch(monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_GOAL_STATE_MAX_AGE_SEC", "60")
    store = ActiveGoalStore()
    stale = _goal().model_copy(update={"ts": datetime.now(timezone.utc) - timedelta(seconds=120)})
    store.update_from_goal(stale)
    assert store.current() is None


def test_active_goal_store_fresh_goal_survives_staleness_check(monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_GOAL_STATE_MAX_AGE_SEC", "3600")
    store = ActiveGoalStore()
    store.update_from_goal(_goal())
    assert store.current() is not None


def test_active_goal_store_clear() -> None:
    store = ActiveGoalStore()
    store.update_from_goal(_goal())
    store.clear()
    assert store.current() is None


def test_active_goal_store_malformed_goal_is_noop() -> None:
    store = ActiveGoalStore()

    class _Bad:
        proposal_status = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    store.update_from_goal(_Bad())  # type: ignore[arg-type]
    assert store.current() is None
