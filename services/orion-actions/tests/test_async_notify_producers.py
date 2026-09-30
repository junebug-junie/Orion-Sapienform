from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.main import (
    ACTION_DAILY_METACOG_V1,
    ACTION_DAILY_PULSE_V1,
    _build_post_persist_journal_message_payload,
    _daily_notify_request,
    _publish_daily_outputs,
    _publish_workflow_attention_signal,
    _send_orion_async_message,
    _send_pending_attention,
    settings,
)
from app.workflow_schedule_store import WorkflowScheduleStore
from orion.journaler.schemas import JournalEntryWriteV1
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
        include_email_channel=True,
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
        include_email_channel=True,
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


def test_daily_notify_request_includes_email_channel_when_enabled() -> None:
    req = _daily_notify_request(
        event_kind="orion.daily.pulse",
        title="Orion — Daily Pulse",
        dedupe_key="dedupe-key",
        correlation_id="corr-email-on",
        payload={"date": "2026-04-25"},
        include_email_channel=True,
    )
    assert req.channels_requested == ["email"]


def test_daily_notify_request_omits_email_channel_when_disabled() -> None:
    req = _daily_notify_request(
        event_kind="orion.daily.metacog",
        title="Orion — Daily Metacog",
        dedupe_key="dedupe-key",
        correlation_id="corr-email-off",
        payload={"date": "2026-04-25"},
        include_email_channel=False,
    )
    assert req.channels_requested is None


def test_daily_notify_request_uses_full_body_and_payload_fingerprint_dedupe() -> None:
    payload = {"date": "2026-04-25", "tone": "current"}
    req = _daily_notify_request(
        event_kind="orion.daily.pulse",
        title="Orion — Daily Pulse",
        dedupe_key="dedupe-key",
        correlation_id="corr-fingerprint",
        payload=payload,
        include_email_channel=True,
    )
    assert req.body_text == req.body_md
    assert str(req.body_text or "").startswith("## Orion — Daily Pulse")
    assert req.dedupe_key is not None
    assert req.dedupe_key.startswith("dedupe-key:")

    req_new_payload = _daily_notify_request(
        event_kind="orion.daily.pulse",
        title="Orion — Daily Pulse",
        dedupe_key="dedupe-key",
        correlation_id="corr-fingerprint-2",
        payload={"date": "2026-04-25", "tone": "updated"},
        include_email_channel=True,
    )
    assert req_new_payload.dedupe_key != req.dedupe_key


def test_publish_daily_outputs_requires_both_async_flags(monkeypatch) -> None:
    notify = _FakeNotify()
    monkeypatch.setattr(settings, "actions_preserve_generic_notify_enabled", True)
    monkeypatch.setattr(settings, "actions_async_messages_enabled", True)
    monkeypatch.setattr(settings, "actions_daily_async_messages_enabled", False)
    req = _daily_notify_request(
        event_kind="orion.daily.pulse",
        title="Orion — Daily Pulse",
        dedupe_key="dedupe-key",
        correlation_id="corr-async-flags",
        payload={"date": "2026-04-25"},
        include_email_channel=True,
    )

    _publish_daily_outputs(
        notify=notify,
        action_name=ACTION_DAILY_PULSE_V1,
        title="Orion — Daily Pulse",
        preview_text="preview",
        full_text="full",
        notify_req=req,
        correlation_id="corr-async-flags",
    )
    assert len(notify.chat_calls) == 0

    monkeypatch.setattr(settings, "actions_daily_async_messages_enabled", True)
    _publish_daily_outputs(
        notify=notify,
        action_name=ACTION_DAILY_PULSE_V1,
        title="Orion — Daily Pulse",
        preview_text="preview",
        full_text="full",
        notify_req=req,
        correlation_id="corr-async-flags",
    )
    assert len(notify.chat_calls) == 1


def test_post_persist_journal_message_payload_builder() -> None:
    """`_build_post_persist_journal_message_payload` is still used (by
    `_dispatch_journal_notifications`, see test_journal_actions.py) to build the
    shared message content for both the in-app and email channels. The old
    scheduler-daily-specific and post-persist-email-specific builder/gate helpers
    (`_is_scheduler_daily_journal`, `_build_scheduler_daily_journal_*`,
    `_should_email_persisted_journal`, `_build_post_persist_journal_email_request`)
    were retired in favor of `orion.journaler.dispatch_registry.resolve_policy` --
    see services/orion-actions/tests/test_journal_actions.py for the regression
    coverage of that unification."""
    entry = JournalEntryWriteV1(
        entry_id="entry-post-1",
        author="orion",
        mode="manual",
        title="Journal Pass Entry",
        body="Captured insight text.",
        source_kind="manual",
        source_ref="manual-1",
        correlation_id="corr-post-1",
    )
    message_payload = _build_post_persist_journal_message_payload(entry=entry, correlation_id="corr-post-1")
    assert message_payload["title"] == "Orion — Journal Pass"
    assert "Journal Pass Entry" in message_payload["preview_text"]
    assert "Entry ID" in message_payload["full_text"]


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

    # Attention pages when the retry budget is spent, not on the first failure;
    # a 1-attempt budget makes a single failure reach that condition.
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"), max_dispatch_attempts=1)
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

    # Attention pages when the retry budget is spent, not on the first failure;
    # a 1-attempt budget makes a single failure reach that condition.
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"), max_dispatch_attempts=1)
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


def test_daily_json_emails_are_retired_by_default_but_in_app_delivery_remains(monkeypatch) -> None:
    """Retired 2026-09-30: Daily Pulse / Daily Metacog no longer request email.

    When generation is enabled, in-app delivery (Hub notification via notify.send +
    async chat message) must still happen. Generation itself is paused by default
    since 2026-09-30 (see test_daily_pulse_and_metacog_generation_paused_by_default).
    notify only emails an info-severity request when channels_requested contains
    "email" (orion-notify email_delivery.should_send_email), so None == no email.
    """
    from pathlib import Path

    from app.settings import Settings

    monkeypatch.delenv("ACTIONS_DAILY_EMAIL_ENABLED", raising=False)
    assert Settings(_env_file=None).actions_daily_email_enabled is False

    env_example = Path(__file__).resolve().parents[1] / ".env_example"
    lines = [ln.strip() for ln in env_example.read_text(encoding="utf-8").splitlines()]
    assert "ACTIONS_DAILY_EMAIL_ENABLED=false" in lines

    monkeypatch.setattr(settings, "actions_daily_email_enabled", False)
    monkeypatch.setattr(settings, "actions_preserve_generic_notify_enabled", True)
    monkeypatch.setattr(settings, "actions_async_messages_enabled", True)
    monkeypatch.setattr(settings, "actions_daily_async_messages_enabled", True)
    for action_name, event_kind, title in (
        (ACTION_DAILY_PULSE_V1, "orion.daily.pulse", "Orion — Daily Pulse"),
        (ACTION_DAILY_METACOG_V1, "orion.daily.metacog", "Orion — Daily Metacog"),
    ):
        notify = _FakeNotify()
        req = _daily_notify_request(
            event_kind=event_kind,
            title=title,
            dedupe_key="dedupe-key",
            correlation_id="corr-retired",
            payload={"date": "2026-09-30"},
            include_email_channel=settings.actions_daily_email_enabled,
        )
        assert req.channels_requested is None
        assert req.severity == "info"
        _publish_daily_outputs(
            notify=notify,
            action_name=action_name,
            title=title,
            preview_text="preview",
            full_text="full",
            notify_req=req,
            correlation_id="corr-retired",
        )
        assert len(notify.send_calls) == 1
        assert not notify.send_calls[0].channels_requested
        assert len(notify.chat_calls) == 1


def test_daily_pulse_and_metacog_generation_paused_by_default(monkeypatch) -> None:
    # Paused 2026-09-30 until there is a real consumer; the only one was self-experiments skill probes.
    from pathlib import Path

    from app.settings import Settings

    monkeypatch.delenv("ACTIONS_DAILY_PULSE_ENABLED", raising=False)
    monkeypatch.delenv("ACTIONS_DAILY_METACOG_ENABLED", raising=False)

    s = Settings(_env_file=None)
    assert s.actions_daily_pulse_enabled is False
    assert s.actions_daily_metacog_enabled is False
    env_example = Path(__file__).resolve().parents[1] / ".env_example"
    lines = [ln.strip() for ln in env_example.read_text(encoding="utf-8").splitlines()]
    assert "ACTIONS_DAILY_PULSE_ENABLED=false" in lines
    assert "ACTIONS_DAILY_METACOG_ENABLED=false" in lines
