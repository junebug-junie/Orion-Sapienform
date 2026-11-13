"""Hub outreach goes through /chat/message but must stay recognisable in the tray.

Before 2026-10-29 Hub published outreach straight to orion:notify:in_app, so no
notify_requests row existed for any of it (24 sends on 2026-10-28, 0 rows) and
a Hub restart wiped it. Now Hub posts it here. The caller-chosen
`notification_type` must survive into the live in-app event, or Hub would toast
every outreach a second time and could not pin it as an Orion message.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

SERVICE_DIR = str(Path(__file__).resolve().parents[1])
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("SERVICE_NAME", "orion-notify")

from orion.schemas.notify import ChatMessageNotification  # noqa: E402


class _Bus:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, channel, env) -> None:
        self.published.append((channel, env))


def _message(**kw) -> ChatMessageNotification:
    base = dict(
        message_id=uuid4(),
        source_service="orion-hub",
        session_id="sess-1",
        correlation_id="corr-1",
        title="Orion reached out",
        preview_text="hello",
        full_text="hello there",
        tags=["endogenous_outreach"],
    )
    base.update(kw)
    return ChatMessageNotification(**base)


def _in_app_payload(msg: ChatMessageNotification) -> dict:
    from app.main import _chat_message_to_notification, _publish_in_app_event

    notify_payload = _chat_message_to_notification(msg)
    bus = _Bus()
    asyncio.run(_publish_in_app_event(bus, str(notify_payload.notification_id), notify_payload, "delivered"))
    assert len(bus.published) == 1
    return bus.published[0][1].payload


def test_caller_notification_type_reaches_the_in_app_event() -> None:
    msg = _message(notification_type="endogenous_outreach")
    payload = _in_app_payload(msg)
    assert payload["notification_type"] == "endogenous_outreach"
    # Still the chat-message kind: sql-writer fills message_* columns (the
    # durable lifecycle) only for this kind.
    assert payload["event_kind"] == "orion.chat.message"
    assert payload["message_id"] == str(msg.message_id)
    assert payload["session_id"] == "sess-1"


def test_without_a_caller_type_the_kind_decides() -> None:
    payload = _in_app_payload(_message())
    assert payload["notification_type"] == "chat_message"


def test_the_type_is_persisted_in_the_request_context() -> None:
    from app.main import _chat_message_to_notification

    req = _chat_message_to_notification(_message(notification_type="endogenous_outreach"))
    assert req.context["notification_type"] == "endogenous_outreach"
    assert req.context["message_id"]
    req_plain = _chat_message_to_notification(_message())
    assert "notification_type" not in req_plain.context


def test_blank_type_is_ignored() -> None:
    from app.main import _context_notification_type

    assert _context_notification_type({"notification_type": "  "}) is None
    assert _context_notification_type(None) is None
    assert _context_notification_type({"notification_type": "x"}) == "x"


def test_chat_messages_proxy_passes_the_tag_filter(monkeypatch) -> None:
    import app.main as main

    seen = {}

    async def fake_proxy_get(path, params=None):
        seen["path"] = path
        seen["params"] = params
        return []

    monkeypatch.setattr(main, "_proxy_get", fake_proxy_get)
    monkeypatch.setattr(main.settings, "API_TOKEN", None, raising=False)
    asyncio.run(
        main.list_chat_messages(
            limit=10, status="open", session_id=None, tag="endogenous_outreach", x_orion_notify_token=None
        )
    )
    assert seen["path"] == "/chat/messages"
    assert seen["params"]["status"] == "open"
    assert seen["params"]["tag"] == "endogenous_outreach"


def test_chat_message_without_a_bus_is_refused_not_faked(monkeypatch) -> None:
    """A 200 with no bus meant nothing was saved or delivered, yet Hub skipped
    its fallback. 503 lets the caller fall back."""
    import pytest
    from fastapi import HTTPException
    from types import SimpleNamespace

    import app.main as main

    monkeypatch.setattr(main.settings, "API_TOKEN", None, raising=False)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(bus=None)))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.chat_message(_message(notification_type="endogenous_outreach"), request, None))
    assert exc.value.status_code == 503
