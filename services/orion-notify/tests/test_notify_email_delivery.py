import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app import main
from orion.schemas.notify import NotificationRequest


class DummyTransport:
    def __init__(self, should_raise: bool = False) -> None:
        self.calls = []
        self.should_raise = should_raise

    def send(self, payload: NotificationRequest) -> None:
        self.calls.append(payload)
        if self.should_raise:
            raise RuntimeError("smtp down")


class DummyBus:
    pass


@pytest.mark.asyncio
async def test_startup_creates_email_transport_when_configured(monkeypatch):
    async def fake_init_bus():
        return None

    monkeypatch.setattr(main, "_init_bus", fake_init_bus)
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_SMTP_PORT", 587)
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_SMTP_USERNAME", "user")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_SMTP_PASSWORD", "pass")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_USE_TLS", True)
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_FROM", "from@example.com")
    monkeypatch.setattr(main.settings, "NOTIFY_EMAIL_TO", "to1@example.com,to2@example.com")

    await main.on_startup()

    transport = main.app.state.email_transport
    assert transport is not None
    assert transport.smtp_host == "smtp.example.com"
    assert transport.default_to == ["to1@example.com", "to2@example.com"]


@pytest.mark.asyncio
async def test_notify_sends_email_when_channel_requested(monkeypatch):
    sent = DummyTransport()
    published = {"in_app": 0, "persistence": 0}

    async def fake_in_app(*args, **kwargs):
        published["in_app"] += 1

    async def fake_persist(*args, **kwargs):
        published["persistence"] += 1

    monkeypatch.setattr(main.settings, "NOTIFY_IN_APP_ENABLED", True)
    monkeypatch.setattr(main, "_publish_in_app_event", fake_in_app)
    monkeypatch.setattr(main, "_publish_persistence_event", fake_persist)
    monkeypatch.setattr(main.asyncio, "create_task", lambda coro: asyncio.create_task(coro))

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(bus=DummyBus(), email_transport=sent)))
    payload = NotificationRequest(
        source_service="svc",
        event_kind="evt",
        severity="info",
        title="hello",
        channels_requested=["email"],
    )

    result = await main.notify(payload, request)
    await asyncio.sleep(0)

    assert result.status == "queued"
    assert len(sent.calls) == 1
    assert published["in_app"] == 1
    assert published["persistence"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("severity", ["error", "critical"])
async def test_notify_sends_email_for_error_and_critical(monkeypatch, severity):
    sent = DummyTransport()

    async def fake_in_app(*args, **kwargs):
        return None

    async def fake_persist(*args, **kwargs):
        return None

    monkeypatch.setattr(main.settings, "NOTIFY_IN_APP_ENABLED", True)
    monkeypatch.setattr(main, "_publish_in_app_event", fake_in_app)
    monkeypatch.setattr(main, "_publish_persistence_event", fake_persist)
    monkeypatch.setattr(main.asyncio, "create_task", lambda coro: asyncio.create_task(coro))

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(bus=DummyBus(), email_transport=sent)))
    payload = NotificationRequest(
        source_service="svc",
        event_kind="evt",
        severity=severity,
        title="hello",
    )

    await main.notify(payload, request)
    await asyncio.sleep(0)

    assert len(sent.calls) == 1


@pytest.mark.asyncio
async def test_notify_does_not_send_email_for_info_without_email_channel(monkeypatch):
    sent = DummyTransport()

    async def fake_in_app(*args, **kwargs):
        return None

    async def fake_persist(*args, **kwargs):
        return None

    monkeypatch.setattr(main.settings, "NOTIFY_IN_APP_ENABLED", True)
    monkeypatch.setattr(main, "_publish_in_app_event", fake_in_app)
    monkeypatch.setattr(main, "_publish_persistence_event", fake_persist)
    monkeypatch.setattr(main.asyncio, "create_task", lambda coro: asyncio.create_task(coro))

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(bus=DummyBus(), email_transport=sent)))
    payload = NotificationRequest(
        source_service="svc",
        event_kind="evt",
        severity="info",
        title="hello",
    )

    await main.notify(payload, request)
    await asyncio.sleep(0)

    assert sent.calls == []


@pytest.mark.asyncio
async def test_notify_publishes_even_if_smtp_send_fails(monkeypatch):
    sent = DummyTransport(should_raise=True)
    published = {"in_app": 0, "persistence": 0}

    async def fake_in_app(*args, **kwargs):
        published["in_app"] += 1

    async def fake_persist(*args, **kwargs):
        published["persistence"] += 1

    monkeypatch.setattr(main.settings, "NOTIFY_IN_APP_ENABLED", True)
    monkeypatch.setattr(main, "_publish_in_app_event", fake_in_app)
    monkeypatch.setattr(main, "_publish_persistence_event", fake_persist)
    monkeypatch.setattr(main.asyncio, "create_task", lambda coro: asyncio.create_task(coro))

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(bus=DummyBus(), email_transport=sent)))
    payload = NotificationRequest(
        source_service="svc",
        event_kind="evt",
        severity="critical",
        title="hello",
        notification_id=uuid4(),
    )

    result = await main.notify(payload, request)
    await asyncio.sleep(0)

    assert result.status == "queued"
    assert len(sent.calls) == 1
    assert published["in_app"] == 1
    assert published["persistence"] == 1
