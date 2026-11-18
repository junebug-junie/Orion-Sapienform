"""Orion's outreach messages are durable until Juniper dismisses them.

Verified live 2026-10-28: 24 outreach sends, 0 notify_requests rows. Outreach
published straight to orion:notify:in_app, which only Hub's in-memory cache
(200 items) reads, so a Hub restart or alert noise erased it. Outreach now goes
through orion-notify's /chat/message (durable row keyed by the chat message
id). The old direct publish stays only as the fallback when orion-notify is
unreachable, so the live tray never loses a message.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

from scripts.endogenous_outreach import OUTREACH_TAG, EndogenousOutreach


async def _never_fires(*_a, **_kw):
    return None


def _outreach() -> EndogenousOutreach:
    return EndogenousOutreach(
        enabled=True,
        tick_interval_sec=300.0,
        min_cooldown_sec=0.0,
        daily_cap=4,
        quiet_start_hour=-1,
        quiet_end_hour=-1,
        timeout_sec=5.0,
        agent_lane_timeout_sec=5.0,
        notify_channel="orion:notify:in_app",
        fallback_session_id="orion_outreach",
        trigger_evaluator=_never_fires,
    )


class _Bus:
    def __init__(self) -> None:
        self.published: list = []
        self.enabled = True

    async def publish(self, channel, envelope) -> None:
        self.published.append((channel, envelope))


class _Client:
    def __init__(self, *, ok=True, raises=None) -> None:
        self.ok = ok
        self.raises = raises
        self.calls: list = []

    def chat_message(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return SimpleNamespace(ok=self.ok, notification_id=uuid4() if self.ok else None, detail=None if self.ok else "502")


def _send(outreach, **kw):
    params = dict(
        text="the cabinet ran hot while you were out",
        session_id="sess-abc",
        correlation_id=str(uuid4()),
        message_id=str(uuid4()),
    )
    params.update(kw)
    asyncio.run(outreach._publish_notification(**params))
    return params


def test_outreach_goes_through_notify_and_skips_the_direct_publish() -> None:
    outreach = _outreach()
    bus = _Bus()
    outreach._bus = bus
    client = _Client()
    outreach.notify_client = client

    sent = _send(outreach)

    # orion-notify publishes the live event itself; a second one would show
    # the message twice in the tray.
    assert bus.published == []
    assert len(client.calls) == 1
    call = client.calls[0]
    # Same id as the chat_message row, so Reply can find the message in chat.
    assert call["message_id"] == UUID(sent["message_id"])
    assert call["session_id"] == "sess-abc"
    assert call["correlation_id"] == sent["correlation_id"]
    assert call["notification_type"] == OUTREACH_TAG
    assert OUTREACH_TAG in call["tags"]
    assert call["full_text"] == sent["text"]
    assert call["source_service"] == "orion-hub"


def test_custom_tags_still_carry_the_outreach_tag() -> None:
    outreach = _outreach()
    outreach._bus = _Bus()
    client = _Client()
    outreach.notify_client = client

    _send(outreach, tags=["reading_finding"])

    assert client.calls[0]["tags"] == ["reading_finding", OUTREACH_TAG]


def test_long_text_is_previewed_within_the_schema_limit() -> None:
    from orion.schemas.notify import ChatMessageNotification

    outreach = _outreach()
    outreach._bus = _Bus()
    client = _Client()
    outreach.notify_client = client
    long_text = "x" * 1000

    _send(outreach, text=long_text)

    call = client.calls[0]
    assert len(call["preview_text"]) == 280
    assert call["full_text"] == long_text
    # The real request model must accept what we send.
    ChatMessageNotification(
        source_service=call["source_service"],
        session_id=call["session_id"],
        preview_text=call["preview_text"],
        full_text=call["full_text"],
        message_id=call["message_id"],
        notification_type=call["notification_type"],
    )


def test_notify_refusal_falls_back_to_the_direct_publish() -> None:
    outreach = _outreach()
    bus = _Bus()
    outreach._bus = bus
    outreach.notify_client = _Client(ok=False)

    _send(outreach)

    assert len(bus.published) == 1
    payload = bus.published[0][1].payload
    assert payload["notification_type"] == OUTREACH_TAG


def test_notify_exception_falls_back_and_does_not_raise() -> None:
    outreach = _outreach()
    bus = _Bus()
    outreach._bus = bus
    outreach.notify_client = _Client(raises=RuntimeError("boom"))

    _send(outreach)

    assert len(bus.published) == 1


def test_non_outreach_types_do_not_become_durable_orion_messages() -> None:
    # Collapse-mirror replies answer Juniper while she is there; they are not
    # unsolicited outreach and must not pin in the tray.
    outreach = _outreach()
    bus = _Bus()
    outreach._bus = bus
    client = _Client()
    outreach.notify_client = client

    _send(outreach, notification_type="collapse_mirror_reply")

    assert client.calls == []
    assert len(bus.published) == 1
    assert bus.published[0][1].payload["notification_type"] == "collapse_mirror_reply"


def test_no_client_keeps_the_old_behaviour() -> None:
    outreach = _outreach()
    bus = _Bus()
    outreach._bus = bus

    _send(outreach)

    assert len(bus.published) == 1
    assert bus.published[0][1].payload["event_kind"] == "hub.endogenous_outreach.v1"


def test_real_notify_client_sends_message_id_and_type(monkeypatch) -> None:
    """The shared NotifyClient must put both new fields on the wire."""
    import orion.notify.client as client_mod
    from orion.notify.client import NotifyClient

    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"notification_id": str(uuid4())}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    mid = uuid4()
    accepted = NotifyClient("http://notify").chat_message(
        session_id="s",
        preview_text="p",
        message_id=mid,
        notification_type=OUTREACH_TAG,
    )
    assert accepted.ok is True
    assert captured["url"] == "http://notify/chat/message"
    assert captured["json"]["message_id"] == str(mid)
    assert captured["json"]["notification_type"] == OUTREACH_TAG
