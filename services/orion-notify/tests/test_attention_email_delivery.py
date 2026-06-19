import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import main
from orion.schemas.notify import ChatAttentionRequest, NotificationRequest


class DummyTransport:
    def __init__(self) -> None:
        self.calls: list[NotificationRequest] = []

    def send(self, payload: NotificationRequest) -> None:
        self.calls.append(payload)


class DummyBus:
    pass


class DummyPolicy:
    def evaluate(self, payload, now):
        return SimpleNamespace(ack_deadline_minutes=60, escalation_channels=["email"])


def _noop_create_task(coro):
    if asyncio.iscoroutine(coro):
        coro.close()
    return None


@pytest.mark.asyncio
async def test_attention_request_sends_email_for_critical(monkeypatch):
    sent = DummyTransport()

    async def fake_in_app(*args, **kwargs):
        return None

    async def fake_persist(*args, **kwargs):
        return None

    monkeypatch.setattr(main.settings, "NOTIFY_IN_APP_ENABLED", True)
    monkeypatch.setattr(main, "_publish_in_app_event", fake_in_app)
    monkeypatch.setattr(main, "_publish_persistence_event", fake_persist)
    monkeypatch.setattr(main.asyncio, "create_task", _noop_create_task)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                bus=DummyBus(),
                email_transport=sent,
                policy=DummyPolicy(),
            )
        )
    )
    payload = ChatAttentionRequest(
        source_service="orion-mesh-guardian",
        reason="[Orion mesh] landing-pad — observe_only",
        severity="critical",
        message="mesh health: landing-pad observe-only",
        require_ack=True,
        context={"service_id": "landing-pad", "event": "observe_only"},
    )

    await main.attention_request(payload, request)
    await asyncio.sleep(0)

    assert len(sent.calls) == 1
    assert sent.calls[0].severity == "critical"


@pytest.mark.asyncio
async def test_attention_request_skips_immediate_email_for_error(monkeypatch):
    sent = DummyTransport()

    async def fake_in_app(*args, **kwargs):
        return None

    async def fake_persist(*args, **kwargs):
        return None

    monkeypatch.setattr(main.settings, "NOTIFY_IN_APP_ENABLED", True)
    monkeypatch.setattr(main, "_publish_in_app_event", fake_in_app)
    monkeypatch.setattr(main, "_publish_persistence_event", fake_persist)
    monkeypatch.setattr(main.asyncio, "create_task", _noop_create_task)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                bus=DummyBus(),
                email_transport=sent,
                policy=DummyPolicy(),
            )
        )
    )
    payload = ChatAttentionRequest(
        source_service="orion-mesh-guardian",
        reason="attention_request",
        severity="error",
        message="mesh health: landing-pad unhealthy confirmed",
        require_ack=True,
        context={"service_id": "landing-pad", "event": "unhealthy_confirmed"},
    )

    await main.attention_request(payload, request)
    await asyncio.sleep(0)

    assert sent.calls == []
