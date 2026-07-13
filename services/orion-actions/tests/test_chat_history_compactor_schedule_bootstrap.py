from __future__ import annotations

from pathlib import Path

from orion.schemas.workflow_execution import (
    WorkflowScheduleManageRequestV1,
    WorkflowScheduleUpdatePatchV1,
)

from app.workflow_schedule_bootstrap import ensure_chat_history_compactor_daily_schedule
from app.workflow_schedule_store import WorkflowScheduleStore


def test_chat_history_compactor_schedule_bootstrap_creates_once(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    first = ensure_chat_history_compactor_daily_schedule(store)
    assert first is not None
    assert first.workflow_id == "chat_history_compactor_pass"
    assert first.execution_policy.schedule is not None
    assert first.execution_policy.schedule.kind == "recurring"
    assert first.execution_policy.schedule.cadence == "daily"
    assert first.execution_policy.schedule.hour_local == 6
    assert first.execution_policy.schedule.minute_local == 0
    assert first.execution_policy.schedule.timezone == "America/Denver"
    assert first.workflow_request.get("window_mode") == "day"

    second = ensure_chat_history_compactor_daily_schedule(store)
    assert second is None
    active = [s for s in store.list_schedules() if s.workflow_id == "chat_history_compactor_pass"]
    assert len(active) == 1


def test_bootstrap_does_not_resurrect_cancelled_schedule(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_chat_history_compactor_daily_schedule(store)
    assert record is not None

    resp = store.apply_management(
        WorkflowScheduleManageRequestV1(
            operation="cancel",
            request_id="test-cancel",
            schedule_id=record.schedule_id,
        )
    )
    assert resp.ok is True

    # Simulated restart: bootstrap must respect the operator's cancel.
    assert ensure_chat_history_compactor_daily_schedule(store) is None
    active = [s for s in store.list_schedules() if s.workflow_id == "chat_history_compactor_pass"]
    assert active == []


def test_bootstrap_does_not_duplicate_operator_edited_schedule(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_chat_history_compactor_daily_schedule(store)
    assert record is not None

    resp = store.apply_management(
        WorkflowScheduleManageRequestV1(
            operation="update",
            request_id="test-update",
            schedule_id=record.schedule_id,
            patch=WorkflowScheduleUpdatePatchV1(hour_local=7),
        )
    )
    assert resp.ok is True

    # Simulated restart: the edited schedule counts as existing; no 06:00 twin.
    assert ensure_chat_history_compactor_daily_schedule(store) is None
    active = [s for s in store.list_schedules() if s.workflow_id == "chat_history_compactor_pass"]
    assert len(active) == 1
    assert active[0].execution_policy.schedule.hour_local == 7
