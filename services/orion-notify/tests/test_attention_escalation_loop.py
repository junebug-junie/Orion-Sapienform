from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.attention_escalation import run_attention_escalation_once


class DummyTransport:
    def __init__(self) -> None:
        self.calls = []

    def send(self, payload) -> None:
        self.calls.append(payload)


@pytest.mark.asyncio
async def test_escalation_sends_email_for_stale_error_attention():
    sent = DummyTransport()
    old = datetime.now(timezone.utc) - timedelta(minutes=90)
    row = {
        "attention_id": "att-1",
        "severity": "error",
        "require_ack": True,
        "acked_at": None,
        "escalated_at": None,
        "created_at": old.isoformat(),
        "ack_deadline_minutes": 60,
        "reason": "attention_request",
        "message": "mesh health: landing-pad unhealthy confirmed",
        "source_service": "orion-mesh-guardian",
        "context": {"escalation_channels": ["email"], "service_id": "landing-pad"},
        "correlation_id": "corr-1",
        "session_id": None,
    }
    proxy_get = AsyncMock(return_value=[row])
    proxy_post = AsyncMock(return_value={"status": "escalated"})
    policy = SimpleNamespace(
        evaluate=lambda payload, now: SimpleNamespace(ack_deadline_minutes=60, escalation_channels=["email"])
    )

    count = await run_attention_escalation_once(
        email_transport=sent,
        policy=policy,
        proxy_get=proxy_get,
        proxy_post=proxy_post,
        hub_url_base="http://hub.example",
    )

    assert count == 1
    assert len(sent.calls) == 1
    assert sent.calls[0].event_kind == "orion.chat.attention.escalation"
    proxy_post.assert_awaited_once_with("/attention/att-1/escalate", {})


@pytest.mark.asyncio
async def test_escalation_marks_before_send_even_if_smtp_fails():
    class FailingTransport:
        def send(self, payload) -> None:
            raise RuntimeError("smtp down")

    old = datetime.now(timezone.utc) - timedelta(minutes=90)
    row = {
        "attention_id": "att-3",
        "severity": "error",
        "require_ack": True,
        "acked_at": None,
        "escalated_at": None,
        "created_at": old.isoformat(),
        "ack_deadline_minutes": 60,
        "reason": "attention_request",
        "message": "mesh health: landing-pad unhealthy confirmed",
        "source_service": "orion-mesh-guardian",
        "context": {"escalation_channels": ["email"]},
    }
    proxy_get = AsyncMock(return_value=[row])
    proxy_post = AsyncMock(return_value={"status": "escalated"})
    policy = SimpleNamespace(
        evaluate=lambda payload, now: SimpleNamespace(ack_deadline_minutes=60, escalation_channels=["email"])
    )

    count = await run_attention_escalation_once(
        email_transport=FailingTransport(),
        policy=policy,
        proxy_get=proxy_get,
        proxy_post=proxy_post,
        hub_url_base="",
    )

    assert count == 1
    proxy_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_escalation_skips_critical_attention_rows():
    sent = DummyTransport()
    old = datetime.now(timezone.utc) - timedelta(minutes=90)
    row = {
        "attention_id": "att-2",
        "severity": "critical",
        "require_ack": True,
        "acked_at": None,
        "escalated_at": None,
        "created_at": old.isoformat(),
        "ack_deadline_minutes": 60,
        "reason": "attention_request",
        "message": "mesh health: landing-pad observe-only",
        "source_service": "orion-mesh-guardian",
        "context": {"escalation_channels": ["email"]},
    }
    proxy_get = AsyncMock(return_value=[row])
    proxy_post = AsyncMock()
    policy = SimpleNamespace(
        evaluate=lambda payload, now: SimpleNamespace(ack_deadline_minutes=60, escalation_channels=["email"])
    )

    count = await run_attention_escalation_once(
        email_transport=sent,
        policy=policy,
        proxy_get=proxy_get,
        proxy_post=proxy_post,
        hub_url_base="",
    )

    assert count == 0
    assert sent.calls == []
    proxy_post.assert_not_awaited()
