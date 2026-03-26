from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from app.main import _publish_workflow_attention_signal, _schedule_attention_notify_request, workflow_schedule_metrics
from app.workflow_schedule_metrics import WorkflowScheduleMetrics
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


def test_attention_notify_integration_dedupe_payload_and_no_spam(tmp_path) -> None:
    metrics = WorkflowScheduleMetrics()
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"), metrics=metrics)
    created = store.upsert_from_dispatch(_dispatch_request(request_id="req-integration", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    assert created is not None
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    store.mark_dispatch_failed(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="boom", now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc))
    signals = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 2, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert len(signals) == 1
    signal = signals[0]
    sent: list = []

    class _Notify:
        def send(self, req):
            sent.append(req)
            return SimpleNamespace(ok=True, detail=None)

    before_entered = workflow_schedule_metrics.get("workflow_schedule_attention_entered_total")
    asyncio.run(_publish_workflow_attention_signal(signal=signal, notify=_Notify()))
    assert workflow_schedule_metrics.get("workflow_schedule_attention_entered_total") == before_entered + 1
    assert len(sent) == 1
    req = sent[0]
    assert req.dedupe_key == f"workflow:schedule:attention:{created.schedule_id}:failing"
    assert req.context["transition"] == "entered"
    assert req.context["condition"] == "failing"
    assert req.context["state"] == "active"
    assert req.context["schedule_id_short"] == created.schedule_id[-8:]

    again = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 3, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert again == []

    store.mark_dispatch_succeeded(
        run_id=claimed[0].run.run_id,
        schedule_id=claimed[0].schedule.schedule_id,
        now_utc=datetime(2026, 3, 25, 7, 4, tzinfo=timezone.utc),
    )
    recovered = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 5, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert len(recovered) == 1
    sent.clear()
    before_recovered = workflow_schedule_metrics.get("workflow_schedule_attention_recovered_total")
    asyncio.run(_publish_workflow_attention_signal(signal=recovered[0], notify=_Notify()))
    assert workflow_schedule_metrics.get("workflow_schedule_attention_recovered_total") == before_recovered + 1
    assert sent[0].dedupe_key == f"workflow:schedule:attention:{created.schedule_id}:recovered"
    assert sent[0].context["transition"] == "recovered"
    assert sent[0].context["condition"] == "ok"


def test_attention_notify_integration_overdue_transition(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    created = store.upsert_from_dispatch(_dispatch_request(request_id="req-overdue", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    assert created is not None
    signals = store.evaluate_attention_signals(
        now_utc=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc),
        overdue_min_seconds=120,
        reminder_cooldown_seconds=9999,
    )
    assert len(signals) == 1
    req = _schedule_attention_notify_request(signal=signals[0], correlation_id="corr-overdue")
    assert req.dedupe_key == f"workflow:schedule:attention:{created.schedule_id}:overdue"
    assert req.context["transition"] == "entered"
    assert req.context["condition"] == "overdue"
    assert req.context["is_overdue"] is True
