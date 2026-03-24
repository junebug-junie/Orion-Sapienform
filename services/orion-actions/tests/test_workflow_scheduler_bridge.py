from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

from orion.schemas.workflow_execution import WorkflowDispatchRequestV1

MODULE_PATH = Path(__file__).resolve().parents[1] / "app" / "workflow_scheduler.py"
SPEC = importlib.util.spec_from_file_location("actions_workflow_scheduler", MODULE_PATH)
assert SPEC and SPEC.loader
workflow_scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow_scheduler
SPEC.loader.exec_module(workflow_scheduler)


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
            "workflow_request": {
                "workflow_id": "journal_pass",
                "execution_policy": {
                    "workflow_id": "journal_pass",
                    "invocation_mode": "scheduled",
                    "notify_on": "completion",
                    "recipient_group": "juniper_primary",
                    "schedule": schedule,
                },
            },
            "execution_policy": {
                "workflow_id": "journal_pass",
                "invocation_mode": "scheduled",
                "notify_on": "completion",
                "recipient_group": "juniper_primary",
                "schedule": schedule,
            },
        }
    )


def test_registers_one_shot_schedule_and_marks_due() -> None:
    schedules = {}
    request = _dispatch_request(request_id="req-1", kind="one_shot")
    workflow_scheduler.register_schedule(schedules=schedules, request=request, now_utc=datetime(2026, 3, 24, 5, 0, tzinfo=timezone.utc))
    assert "req-1" in schedules
    due = workflow_scheduler.due_schedules(schedules, now_utc=datetime(2026, 3, 24, 6, 1, tzinfo=timezone.utc))
    assert len(due) == 1


def test_recurring_schedule_advances_after_dispatch() -> None:
    schedules = {}
    request = _dispatch_request(request_id="req-2", kind="recurring")
    entry = workflow_scheduler.register_schedule(schedules=schedules, request=request, now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    assert entry is not None
    first = entry.next_run_utc
    workflow_scheduler.advance_after_dispatch(schedules=schedules, entry=entry, now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    assert "req-2" in schedules
    assert entry.next_run_utc > first
