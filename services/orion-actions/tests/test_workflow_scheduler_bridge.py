from __future__ import annotations

from datetime import datetime, timezone

from app.workflow_schedule_store import WorkflowScheduleStore
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1


def _dispatch_request(*, request_id: str, kind: str = "one_shot") -> WorkflowDispatchRequestV1:
    schedule = {
        "kind": kind,
        "timezone": "America/Denver",
        "run_at_utc": "2026-03-24T06:00:00Z",
        "label": "test",
    }
    if kind == "recurring":
        schedule = {
            "kind": "recurring",
            "timezone": "America/Denver",
            "cadence": "daily",
            "hour_local": 23,
            "minute_local": 0,
            "label": "nightly",
        }
    return WorkflowDispatchRequestV1.model_validate(
        {
            "request_id": request_id,
            "workflow_id": "journal_pass",
            "workflow_request": {"workflow_id": "journal_pass"},
            "execution_policy": {
                "workflow_id": "journal_pass",
                "invocation_mode": "scheduled",
                "notify_on": "completion",
                "recipient_group": "juniper_primary",
                "schedule": schedule,
            },
        }
    )


def test_claim_due_is_restart_safe(tmp_path) -> None:
    path = tmp_path / "wf-schedules.json"
    store = WorkflowScheduleStore(str(path))
    store.upsert_from_dispatch(_dispatch_request(request_id="req-1", kind="one_shot"))
    assert len(store.claim_due(now_utc=datetime(2026, 3, 24, 6, 1, tzinfo=timezone.utc))) == 1

    reloaded = WorkflowScheduleStore(str(path))
    assert len(reloaded.claim_due(now_utc=datetime(2026, 3, 24, 6, 2, tzinfo=timezone.utc))) == 0


def test_recurring_schedule_advances_after_dispatch(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    store.upsert_from_dispatch(_dispatch_request(request_id="req-2", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    before = store.list_schedules(include_inactive=True)[0].next_run_at
    store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    after = store.list_schedules(include_inactive=True)[0].next_run_at
    assert before is not None and after is not None and after > before


def test_recurring_dispatch_failure_requeues_claimed_slot(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    store.upsert_from_dispatch(_dispatch_request(request_id="req-3", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    assert len(claimed) == 1
    claimed_for = claimed[0].run.metadata.get("claimed_for_run_at")
    assert claimed_for is not None
    store.mark_dispatch_failed(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="downstream failure", now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc))
    reloaded = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    row = reloaded.list_schedules(include_inactive=True)[0]
    assert row.next_run_at is not None
    assert row.next_run_at.isoformat() == claimed_for
