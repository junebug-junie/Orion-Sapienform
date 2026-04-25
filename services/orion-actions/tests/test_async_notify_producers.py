from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.main import (
    ACTION_DAILY_METACOG_V1,
    ACTION_DAILY_PULSE_V1,
    _daily_notify_request,
    _publish_daily_outputs,
    _publish_workflow_attention_signal,
    _send_orion_async_message,
    _send_pending_attention,
    settings,
)
from app.workflow_schedule_store import WorkflowScheduleStore
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1


class _FakeNotify:
    def __init__(self) -> None:
        self.send_calls = []
        self.chat_calls = []
        self.attention_calls = []

    def send(self, request):
        self.send_calls.append(request)
        return SimpleNamespace(ok=True, status="queued", notification_id=None, detail=None)

    def chat_message(self, **kwargs):
        self.chat_calls.append(kwargs)
        return SimpleNamespace(ok=True, notification_id=None, detail=None)

    def attention_request(self, **kwargs):
        self.attention_calls.append(kwargs)
        return SimpleNamespace(ok=True, notification_id=None, detail=None)


def _dispatch_request(*, request_id: str) -> WorkflowDispatchRequestV1:
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
                "schedule": {
                    "kind": "recurring",
                    "timezone": "America/Denver",
                    "cadence": "daily",
                    "hour_local": 23,
                    "minute_local": 0,
                    "label": "nightly",
                },
            },
        }
    )


def test_publish_daily_outputs_calls_chat_message_once_for_daily_pulse(monkeypatch) -> None:
    notify = _FakeNotify()
    monkeypatch.setattr(settings, "actions_async_messages_enabled", True)
    monkeypatch.setattr(settings, "actions_preserve_generic_notify_enabled", True)

    req = _daily_notify_request(
        event_kind="orion.daily.pulse",
        title="Orion — Daily Pulse",
        dedupe_key="dedupe-key",
        correlation_id="corr-1",
        payload={"date": "2026-04-25", "timezone": "America/Denver", "today_focus": "x"},
    )

    _publish_daily_outputs(
        notify=notify,
        action_name=ACTION_DAILY_PULSE_V1,
        title="Orion — Daily Pulse",
        preview_text="preview",
        full_text="full",
        notify_req=req,
        correlation_id="corr-1",
    )

    assert len(notify.send_calls) == 1
    assert len(notify.chat_calls) == 1
    assert notify.chat_calls[0]["title"] == "Orion — Daily Pulse"
    assert notify.chat_calls[0]["correlation_id"] == "corr-1"


def test_publish_daily_outputs_respects_preserve_generic_notify_flag(monkeypatch) -> None:
    notify = _FakeNotify()
    monkeypatch.setattr(settings, "actions_async_messages_enabled", True)
    monkeypatch.setattr(settings, "actions_preserve_generic_notify_enabled", False)

    req = _daily_notify_request(
        event_kind="orion.daily.metacog",
        title="Orion — Daily Metacog",
        dedupe_key="dedupe-key",
        correlation_id="corr-2",
        payload={"date": "2026-04-25", "timezone": "America/Denver", "course_correction": "x"},
    )

    _publish_daily_outputs(
        notify=notify,
        action_name=ACTION_DAILY_METACOG_V1,
        title="Orion — Daily Metacog",
        preview_text="preview",
        full_text="full",
        notify_req=req,
        correlation_id="corr-2",
    )

    assert len(notify.send_calls) == 0
    assert len(notify.chat_calls) == 1


def test_send_pending_attention_routes_to_attention_request() -> None:
    notify = _FakeNotify()
    _send_pending_attention(
        notify=notify,
        reason="Workflow schedule needs attention",
        message="workflow failed",
        severity="error",
        context={"schedule_id": "sched-1"},
        require_ack=True,
    )
    assert len(notify.attention_calls) == 1
    assert notify.attention_calls[0]["severity"] == "error"
    assert notify.attention_calls[0]["context"]["schedule_id"] == "sched-1"


def test_send_orion_async_message_routes_to_chat_message() -> None:
    notify = _FakeNotify()
    _send_orion_async_message(
        notify=notify,
        title="Orion — Daily Pulse",
        preview_text="preview",
        full_text="full",
        correlation_id="corr-3",
    )
    assert len(notify.chat_calls) == 1
    assert notify.chat_calls[0]["title"] == "Orion — Daily Pulse"


def test_workflow_active_attention_calls_attention_request(monkeypatch, tmp_path) -> None:
    notify = _FakeNotify()
    monkeypatch.setattr(settings, "actions_pending_attention_enabled", True)
    monkeypatch.setattr(settings, "actions_async_messages_enabled", True)
    monkeypatch.setattr(settings, "actions_preserve_generic_notify_enabled", True)

    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    created = store.upsert_from_dispatch(_dispatch_request(request_id="req-1"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    store.mark_dispatch_failed(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="boom", now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc))
    signals = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 2, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)

    assert created is not None
    assert len(signals) == 1
    signal = signals[0]
    assert signal.transition == "entered"

    import asyncio

    asyncio.run(_publish_workflow_attention_signal(signal=signal, notify=notify))
    assert len(notify.attention_calls) == 1


def test_workflow_recovered_does_not_call_attention_request(monkeypatch, tmp_path) -> None:
    notify = _FakeNotify()
    monkeypatch.setattr(settings, "actions_pending_attention_enabled", True)
    monkeypatch.setattr(settings, "actions_async_messages_enabled", True)
    monkeypatch.setattr(settings, "actions_preserve_generic_notify_enabled", True)

    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    created = store.upsert_from_dispatch(_dispatch_request(request_id="req-2"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    store.mark_dispatch_failed(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="boom", now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc))
    _ = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 2, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    store.mark_dispatch_succeeded(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, now_utc=datetime(2026, 3, 25, 7, 4, tzinfo=timezone.utc))
    recovered = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 5, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)

    assert created is not None
    assert len(recovered) == 1
    signal = recovered[0]
    assert signal.transition == "recovered"

    import asyncio

    asyncio.run(_publish_workflow_attention_signal(signal=signal, notify=notify))
    assert len(notify.attention_calls) == 0
    assert len(notify.chat_calls) == 1
