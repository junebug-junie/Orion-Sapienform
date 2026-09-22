"""Lend chat lane (chat-burst): Hub holds + emails chat instead of asking Orion.

Covers the hold module itself, the HTTP /api/chat hold, and the gate proxy routes.
The WebSocket hold shares the same two helpers; its loop is not driven here.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

from scripts import chat_lane_lend  # noqa: E402
from scripts.llm_gateway_client import LlmGatewayClientError  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    chat_lane_lend.reset_cache()
    yield
    chat_lane_lend.reset_cache()


# ---------------------------------------------------------------- chat_lane_is_lent


def test_lent_is_false_when_the_gateway_client_raises(monkeypatch):
    """A gateway outage must never block normal chat."""

    async def _boom(route_id):
        raise LlmGatewayClientError("unreachable")

    monkeypatch.setattr(chat_lane_lend, "fetch_route_gate", _boom)
    assert asyncio.run(chat_lane_lend.chat_lane_is_lent()) is False


def test_lent_reads_the_gate_and_caches_a_true(monkeypatch):
    calls = []

    async def _gate(route_id):
        calls.append(route_id)
        return {"route_id": route_id, "open": True, "lends_route": "chat"}

    monkeypatch.setattr(chat_lane_lend, "fetch_route_gate", _gate)
    assert asyncio.run(chat_lane_lend.chat_lane_is_lent()) is True
    assert asyncio.run(chat_lane_lend.chat_lane_is_lent()) is True
    assert calls == ["chat-burst"], "second call inside the TTL must be served from cache"


def test_cache_expires_after_ttl(monkeypatch):
    calls = []

    async def _gate(route_id):
        calls.append(route_id)
        return {"open": False}

    clock = [100.0]
    monkeypatch.setattr(chat_lane_lend.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(chat_lane_lend, "fetch_route_gate", _gate)
    asyncio.run(chat_lane_lend.chat_lane_is_lent())
    clock[0] += chat_lane_lend.CACHE_TTL_SEC + 0.1
    asyncio.run(chat_lane_lend.chat_lane_is_lent())
    assert len(calls) == 2


def test_reset_cache_forces_a_fresh_read(monkeypatch):
    state = {"open": True}
    calls = []

    async def _gate(route_id):
        calls.append(1)
        return dict(state)

    monkeypatch.setattr(chat_lane_lend, "fetch_route_gate", _gate)
    assert asyncio.run(chat_lane_lend.chat_lane_is_lent()) is True
    state["open"] = False
    chat_lane_lend.reset_cache()
    assert asyncio.run(chat_lane_lend.chat_lane_is_lent()) is False
    assert len(calls) == 2


# ------------------------------------------------------ hold_chat_message_for_email


class _FakeAccepted:
    def __init__(self, ok):
        self.ok = ok
        self.detail = None


class _FakeNotifyClient:
    sent = []
    ok = True

    def __init__(self, base_url, api_token=None, timeout=10):
        self.base_url = base_url
        self.api_token = api_token

    def send(self, request):
        _FakeNotifyClient.sent.append(request)
        return _FakeAccepted(_FakeNotifyClient.ok)


@pytest.fixture()
def fake_notify(monkeypatch):
    _FakeNotifyClient.sent = []
    _FakeNotifyClient.ok = True
    monkeypatch.setattr(chat_lane_lend, "NotifyClient", _FakeNotifyClient)
    monkeypatch.setattr(chat_lane_lend.settings, "NOTIFY_BASE_URL", "http://notify.test:7140", raising=False)
    monkeypatch.setattr(chat_lane_lend.settings, "NOTIFY_API_TOKEN", "", raising=False)
    return _FakeNotifyClient


def test_hold_builds_an_email_only_request_and_returns_ok(fake_notify):
    ok = asyncio.run(
        chat_lane_lend.hold_chat_message_for_email(
            text="hello Orion, are you there?",
            session_id="sid-1",
            correlation_id="corr-1",
            mode="brain",
            speaker="juniper",
        )
    )
    assert ok is True
    assert len(fake_notify.sent) == 1
    req = fake_notify.sent[0]
    assert req.channels_requested == ["email"]
    assert req.source_service == "orion-hub"
    assert req.event_kind == "orion.hub.chat.lane_lent"
    assert req.recipient_group == "juniper_primary"
    assert req.dedupe_key == "hub-chat-lent:corr-1"
    assert req.correlation_id == "corr-1"
    assert req.session_id == "sid-1"
    assert "hello Orion, are you there?" in req.body_text
    assert "Mode: brain" in req.body_text
    assert "Speaker: juniper" in req.body_text
    assert req.body_md == req.body_text


def test_hold_returns_false_when_notify_rejects(fake_notify):
    fake_notify.ok = False
    ok = asyncio.run(
        chat_lane_lend.hold_chat_message_for_email(
            text="x", session_id="s", correlation_id="c", mode="brain", speaker="user"
        )
    )
    assert ok is False


def test_hold_returns_false_without_a_notify_url(monkeypatch, fake_notify):
    monkeypatch.setattr(chat_lane_lend.settings, "NOTIFY_BASE_URL", "", raising=False)
    ok = asyncio.run(
        chat_lane_lend.hold_chat_message_for_email(
            text="x", session_id="s", correlation_id="c", mode="brain", speaker="user"
        )
    )
    assert ok is False
    assert fake_notify.sent == []


def test_held_notice_text_marks_a_failed_email():
    assert chat_lane_lend.held_notice_text(True) == chat_lane_lend.HELD_NOTICE_TEXT
    assert chat_lane_lend.held_notice_text(False).endswith("(email delivery failed)")
    assert chat_lane_lend.held_notice_text(False).startswith(chat_lane_lend.HELD_NOTICE_TEXT)


# ------------------------------------------------------------------ HTTP /api/chat


class _Hub:
    """The live Hub modules for one test. Resolved through sys.modules at fixture time,
    NOT the module-level imports above: the Hub conftest purges and re-imports `scripts.*`
    after this file's own imports bind, so patching those top-level names would patch
    orphaned copies and api_routes' lazy `from .chat_lane_lend import ...` would still see
    the real functions (confirmed live: the real gate read ran under a patched test)."""

    def __init__(self, client):
        import importlib

        self.client = client
        self.api_routes = importlib.import_module("scripts.api_routes")
        self.lend = importlib.import_module("scripts.chat_lane_lend")
        self.gateway = importlib.import_module("scripts.llm_gateway_client")


@pytest.fixture()
def hub(monkeypatch):
    from scripts import api_routes
    import scripts.main as hub_main

    class _Bus:
        enabled = True

    monkeypatch.setattr(hub_main, "bus", _Bus(), raising=False)
    monkeypatch.setattr(hub_main, "cortex_client", object(), raising=False)

    async def _ensure_session(session_id, bus):
        return session_id or "sid-test"

    monkeypatch.setattr(api_routes, "ensure_session", _ensure_session)

    app = FastAPI()
    app.include_router(api_routes.router)
    with TestClient(app) as c:
        h = _Hub(c)
        h.lend.reset_cache()
        yield h
        h.lend.reset_cache()


def test_http_chat_is_held_and_never_reaches_cortex_when_lent(monkeypatch, hub):
    client, api_routes, lend = hub.client, hub.api_routes, hub.lend
    published = []
    emailed = []

    async def _lent():
        return True

    async def _hold(**kwargs):
        emailed.append(kwargs)
        return True

    async def _publish(bus, envelopes):
        published.extend(envelopes)

    async def _never(*args, **kwargs):
        raise AssertionError("handle_chat_request must not run while the lane is lent")

    monkeypatch.setattr(lend, "chat_lane_is_lent", _lent)
    monkeypatch.setattr(lend, "hold_chat_message_for_email", _hold)
    monkeypatch.setattr(api_routes, "publish_chat_history", _publish)
    monkeypatch.setattr(api_routes, "handle_chat_request", _never)

    resp = client.post(
        "/api/chat",
        json={"mode": "brain", "user_id": "juniper", "messages": [{"role": "user", "content": "held me"}]},
        headers={"X-Orion-Session-Id": "sid-held"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["held"] is True
    assert body["reason"] == "chat_lane_lent"
    assert body["emailed"] is True
    assert body["text"] == lend.HELD_NOTICE_TEXT
    corr = body["correlation_id"]

    assert len(emailed) == 1
    assert emailed[0]["text"] == "held me"
    assert emailed[0]["session_id"] == "sid-held"
    assert emailed[0]["correlation_id"] == corr
    assert emailed[0]["speaker"] == "juniper"

    assert len(published) == 1, "only the user row is stored; no assistant row for the notice"
    env = published[0]
    assert env.payload.role == "user"
    assert env.payload.content == "held me"
    assert str(env.correlation_id) == corr


def test_http_chat_held_with_no_write_skips_history_but_still_emails(monkeypatch, hub):
    client, api_routes, lend = hub.client, hub.api_routes, hub.lend
    published = []
    emailed = []

    async def _lent():
        return True

    async def _hold(**kwargs):
        emailed.append(kwargs)
        return False

    async def _publish(bus, envelopes):
        published.extend(envelopes)

    monkeypatch.setattr(lend, "chat_lane_is_lent", _lent)
    monkeypatch.setattr(lend, "hold_chat_message_for_email", _hold)
    monkeypatch.setattr(api_routes, "publish_chat_history", _publish)

    resp = client.post("/api/chat", json={"text_input": "fallback text", "no_write": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["emailed"] is False
    assert published == []
    assert emailed[0]["text"] == "fallback text"


def test_http_chat_proceeds_normally_when_not_lent(monkeypatch, hub):
    client, api_routes, lend = hub.client, hub.api_routes, hub.lend
    seen = []

    async def _not_lent():
        return False

    async def _handle(cortex_client, payload, session_id, **kwargs):
        seen.append(payload)
        return {"text": "", "mode": "brain", "correlation_id": "c"}

    monkeypatch.setattr(lend, "chat_lane_is_lent", _not_lent)
    monkeypatch.setattr(api_routes, "handle_chat_request", _handle)
    resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    assert "held" not in resp.json()
    assert len(seen) == 1


# ------------------------------------------------------------------ gate proxies


def test_gate_proxy_returns_404_for_a_non_gated_route(monkeypatch, hub):
    client, gateway = hub.client, hub.gateway
    called = []

    async def _fetch(route_id):
        called.append(route_id)
        return {}

    monkeypatch.setattr(gateway, "fetch_route_gate", _fetch)
    for rid in ("chat", "agent-burst", "nope"):
        assert client.get(f"/api/llm-routes/{rid}/gate").status_code == 404
        r = client.put(f"/api/llm-routes/{rid}/gate", json={"open": True})
        assert r.status_code == 404
        assert r.json()["detail"] == "route_not_operator_gated"
    assert called == [], "the gateway is never asked about a non-gated route"


def test_gate_proxy_get_and_put_pass_through(monkeypatch, hub):
    client, gateway, lend = hub.client, hub.gateway, hub.lend
    puts = []

    async def _fetch(route_id):
        return {"route_id": route_id, "open": False, "changed_at": None, "changed_by": None, "lends_route": "chat"}

    async def _set(route_id, *, open, changed_by):
        puts.append((route_id, open, changed_by))
        return {"route_id": route_id, "open": open, "changed_at": "t", "changed_by": changed_by, "lends_route": "chat"}

    monkeypatch.setattr(gateway, "fetch_route_gate", _fetch)
    monkeypatch.setattr(gateway, "set_route_gate", _set)

    r = client.get("/api/llm-routes/chat-burst/gate")
    assert r.status_code == 200 and r.json()["open"] is False

    # Prime the hold cache with a stale "not lent", then flip: the PUT must drop it.
    lend._cached_lent = False
    lend._cached_at = lend.time.monotonic()
    r = client.put("/api/llm-routes/chat-burst/gate", json={"open": True})
    assert r.status_code == 200, r.text
    assert r.json()["open"] is True
    assert puts == [("chat-burst", True, "hub-ui")]
    assert lend._cached_at is None


def test_gate_proxy_returns_502_when_the_gateway_is_unreachable(monkeypatch, hub):
    client, gateway = hub.client, hub.gateway

    async def _boom(*args, **kwargs):
        # The LIVE module's class -- see _Hub: the top-level import is an orphaned copy
        # and api_routes' `except LlmGatewayClientError` would not catch it.
        raise gateway.LlmGatewayClientError("LLM gateway /routes/chat-burst/gate unreachable")

    monkeypatch.setattr(gateway, "fetch_route_gate", _boom)
    monkeypatch.setattr(gateway, "set_route_gate", _boom)
    assert client.get("/api/llm-routes/chat-burst/gate").status_code == 502
    assert client.put("/api/llm-routes/chat-burst/gate", json={"open": True}).status_code == 502
