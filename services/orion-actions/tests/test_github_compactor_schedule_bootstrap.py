from __future__ import annotations

from pathlib import Path

from orion.schemas.workflow_execution import (
    WorkflowDispatchRequestV1,
    WorkflowScheduleManageRequestV1,
    WorkflowScheduleUpdatePatchV1,
)

from app.workflow_schedule_bootstrap import (
    ensure_chat_history_compactor_daily_schedule,
    ensure_github_compactor_daily_schedule,
)
from app.workflow_schedule_store import WorkflowScheduleStore


def _operator_schedule(store: WorkflowScheduleStore, *, workflow_id: str):
    """A recurring schedule for the same workflow that bootstrap did not create."""
    policy = {
        "workflow_id": workflow_id,
        "invocation_mode": "scheduled",
        "notify_on": "completion",
        "recipient_group": "juniper_primary",
        "schedule": {
            "kind": "recurring",
            "timezone": "America/Denver",
            "cadence": "daily",
            "hour_local": 9,
            "minute_local": 30,
            "label": "operator",
        },
    }
    request = WorkflowDispatchRequestV1.model_validate(
        {
            "request_id": f"operator:{workflow_id}",
            "workflow_id": workflow_id,
            "workflow_request": {"workflow_id": workflow_id, "execution_policy": policy},
            "execution_policy": policy,
        }
    )
    record = store.upsert_from_dispatch(request)
    assert record is not None
    return record



def test_github_compactor_schedule_bootstrap_creates_once(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    first = ensure_github_compactor_daily_schedule(store)
    assert first is not None
    assert first.workflow_id == "github_compactor_pass"
    assert first.execution_policy.schedule is not None
    assert first.execution_policy.schedule.kind == "recurring"
    assert first.execution_policy.schedule.cadence == "daily"
    assert first.execution_policy.schedule.hour_local == 6
    assert first.execution_policy.schedule.minute_local == 10
    assert first.execution_policy.schedule.timezone == "America/Denver"
    assert first.workflow_request.get("window_mode") == "day"

    second = ensure_github_compactor_daily_schedule(store)
    assert second is None
    active = [s for s in store.list_schedules() if s.workflow_id == "github_compactor_pass"]
    assert len(active) == 1


def test_github_bootstrap_does_not_resurrect_cancelled_schedule(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_github_compactor_daily_schedule(store)
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
    assert ensure_github_compactor_daily_schedule(store) is None
    active = [s for s in store.list_schedules() if s.workflow_id == "github_compactor_pass"]
    assert active == []


def test_github_bootstrap_does_not_duplicate_operator_edited_schedule(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_github_compactor_daily_schedule(store)
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

    # Simulated restart: the edited schedule counts as existing; no 06:10 twin.
    assert ensure_github_compactor_daily_schedule(store) is None
    active = [s for s in store.list_schedules() if s.workflow_id == "github_compactor_pass"]
    assert len(active) == 1
    assert active[0].execution_policy.schedule.hour_local == 7


def test_chat_and_github_compactor_bootstraps_coexist_independently(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    chat = ensure_chat_history_compactor_daily_schedule(store)
    github = ensure_github_compactor_daily_schedule(store)
    assert chat is not None
    assert github is not None
    assert chat.schedule_id != github.schedule_id

    # Re-running both bootstraps is a no-op for each independently.
    assert ensure_chat_history_compactor_daily_schedule(store) is None
    assert ensure_github_compactor_daily_schedule(store) is None
    assert len(store.list_schedules()) == 2


def test_github_bootstrap_seeds_failure_only_notifications(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_github_compactor_daily_schedule(store)
    assert record is not None
    assert record.notify_on == "failure"
    assert record.execution_policy.notify_on == "failure"
    # The copy a dispatch forwards to cortex-orch, which is what actually decides
    # whether Juniper is notified.
    assert record.workflow_request["execution_policy"]["notify_on"] == "failure"


def test_github_bootstrap_reconciles_notify_on_for_an_already_seeded_schedule(tmp_path: Path) -> None:
    """Bootstrap is seed-once, so a changed default must still reach live records.

    Every live compactor schedule was seeded with notify_on="completion" and would
    otherwise keep it forever -- one success ping per day, per compactor, for an
    unattended job.
    """
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_github_compactor_daily_schedule(store)
    assert record is not None
    # Simulate the pre-existing live state.
    assert store.set_notify_on(schedule_id=record.schedule_id, notify_on="completion") is True

    assert ensure_github_compactor_daily_schedule(store) is None

    row = [s for s in store.list_schedules() if s.workflow_id == "github_compactor_pass"][0]
    assert row.notify_on == "failure"
    assert row.execution_policy.notify_on == "failure"
    assert row.workflow_request["execution_policy"]["notify_on"] == "failure"


def test_github_bootstrap_reconcile_leaves_operator_schedules_alone(tmp_path: Path) -> None:
    """Only this bootstrap's own record is reconciled.

    An operator-created recurring schedule for the same workflow blocks seeding but
    must not have its notification policy rewritten underneath them.
    """
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_github_compactor_daily_schedule(store)
    assert record is not None
    # Same workflow, different request_id: not ours to reconcile.
    operator = _operator_schedule(store, workflow_id="github_compactor_pass")
    assert store.set_notify_on(schedule_id=operator.schedule_id, notify_on="completion") is False

    assert ensure_github_compactor_daily_schedule(store) is None

    rows = {s.request_id: s for s in store.list_schedules()}
    assert rows[operator.request_id].notify_on == "completion"


def test_github_bootstrap_reconcile_does_not_touch_a_cancelled_schedule(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    record = ensure_github_compactor_daily_schedule(store)
    assert record is not None
    assert store.set_notify_on(schedule_id=record.schedule_id, notify_on="completion") is True
    resp = store.apply_management(
        WorkflowScheduleManageRequestV1(
            operation="cancel",
            request_id="test-cancel-reconcile",
            schedule_id=record.schedule_id,
        )
    )
    assert resp.ok is True

    assert ensure_github_compactor_daily_schedule(store) is None

    row = [s for s in store.list_schedules(include_inactive=True) if s.schedule_id == record.schedule_id][0]
    assert row.state == "cancelled"
    assert row.notify_on == "completion"
