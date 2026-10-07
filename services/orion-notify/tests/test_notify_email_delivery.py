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


def _noop_create_task(coro):
    if asyncio.iscoroutine(coro):
        coro.close()
    return None


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
    monkeypatch.setattr(main.settings, "NOTIFY_ESCALATION_POLL_SECONDS", 0)

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
    monkeypatch.setattr(main.asyncio, "create_task", _noop_create_task)

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
    monkeypatch.setattr(main.asyncio, "create_task", _noop_create_task)

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
    monkeypatch.setattr(main.asyncio, "create_task", _noop_create_task)

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
    monkeypatch.setattr(main.asyncio, "create_task", _noop_create_task)

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


# --------------------------------------------------------------------------
# HTML email with inline (CID) images -- real EmailTransport, captured MIME
# --------------------------------------------------------------------------

import base64 as _b64  # noqa: E402
import re as _re  # noqa: E402

from orion.schemas.notify import NotificationAttachment  # noqa: E402

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _capture_send(monkeypatch, request):
    import smtplib

    from orion.notify.transport import EmailTransport

    captured = []

    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, msg):
            captured.append(msg)
            return {}

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    EmailTransport(
        smtp_host="h", smtp_port=587, smtp_username="u", smtp_password="p",
        use_tls=True, default_from="a@b.c", default_to=["to@example.com"],
    ).send(request)
    assert len(captured) == 1
    return captured[0]


def _req(**kw):
    base = dict(source_service="svc", event_kind="orion.day", severity="info", title="Orion's Day")
    base.update(kw)
    return NotificationRequest(**base)


def _att(name, cid=None, mime="image/png", data=_PNG):
    return NotificationAttachment(
        filename=name, content_base64=_b64.b64encode(data).decode(), mime_type=mime, content_id=cid
    )


def test_plain_text_only_mail_structure_is_unchanged(monkeypatch):
    msg = _capture_send(monkeypatch, _req(body_text="hello"))
    assert msg.get_content_type() == "text/plain"
    assert not msg.is_multipart()
    assert msg.get_content().strip() == "hello"


def test_plain_text_with_attachment_is_still_mixed_and_ignores_content_id(monkeypatch):
    msg = _capture_send(monkeypatch, _req(body_text="hello", attachments=[_att("a.png", cid="img1")]))
    assert msg.get_content_type() == "multipart/mixed"
    parts = msg.get_payload()
    assert [p.get_content_type() for p in parts] == ["text/plain", "image/png"]
    assert parts[1].get_content_disposition() == "attachment"
    assert parts[1]["Content-ID"] is None


def test_html_without_images_is_multipart_alternative(monkeypatch):
    msg = _capture_send(monkeypatch, _req(body_text="fallback", body_html="<h1>Hi</h1>"))
    assert msg.get_content_type() == "multipart/alternative"
    parts = msg.get_payload()
    assert [p.get_content_type() for p in parts] == ["text/plain", "text/html"]
    assert parts[0].get_content().strip() == "fallback"
    assert "<h1>Hi</h1>" in parts[1].get_content()


def test_html_with_inline_images_nests_related_inside_alternative(monkeypatch):
    html = '<p>Today</p><img src="cid:reverie1"><img src="cid:reverie2">'
    msg = _capture_send(
        monkeypatch,
        _req(
            body_text="fallback",
            body_html=html,
            attachments=[_att("r1.png", cid="reverie1"), _att("r2.jpg", cid="<reverie2>", mime="image/jpeg")],
        ),
    )
    assert msg.get_content_type() == "multipart/alternative"
    text_part, related = msg.get_payload()
    assert text_part.get_content_type() == "text/plain"
    assert related.get_content_type() == "multipart/related"
    html_part, *images = related.get_payload()
    assert html_part.get_content_type() == "text/html"
    assert [i.get_content_type() for i in images] == ["image/png", "image/jpeg"]

    # every cid: reference in the HTML resolves to exactly one inline part
    referenced = set(_re.findall(r'cid:([^"\'>\s]+)', html_part.get_content()))
    provided = {i["Content-ID"].strip("<>") for i in images}
    assert referenced == provided == {"reverie1", "reverie2"}
    for i in images:
        assert i.get_content_disposition() == "inline"
        assert i.get_content() == _PNG


def test_html_with_inline_and_regular_attachments_wraps_in_mixed(monkeypatch):
    msg = _capture_send(
        monkeypatch,
        _req(
            body_text="fallback",
            body_html='<img src="cid:x">',
            attachments=[_att("x.png", cid="x"), _att("log.txt", mime="text/plain", data=b"log")],
        ),
    )
    assert msg.get_content_type() == "multipart/mixed"
    alt, attached = msg.get_payload()
    assert alt.get_content_type() == "multipart/alternative"
    assert alt.get_payload()[1].get_content_type() == "multipart/related"
    assert attached.get_content_disposition() == "attachment"
    assert attached.get_filename() == "log.txt"


@pytest.mark.parametrize("html", [
    "<p>" + ("word " * 50000) + "END</p>",
    "<p>" + ("\u00e9" * 100000) + "END</p>",  # single line, non-ASCII, no spaces
    "<p>" + ("x" * 100000) + "END</p>",  # single line, ASCII, no spaces
])
def test_long_html_body_is_not_truncated_and_is_smtp_safe(monkeypatch, html):
    msg = _capture_send(monkeypatch, _req(body_text="fallback", body_html=html))
    assert msg.get_payload()[1].get_content().rstrip().endswith("END</p>")
    # what actually goes on the wire must respect SMTP's 998-octet line limit
    assert max(len(line) for line in msg.as_bytes().split(b"\n")) <= 998


def test_regular_attachment_before_inline_one_still_nests_correctly(monkeypatch):
    msg = _capture_send(
        monkeypatch,
        _req(
            body_text="fallback",
            body_html='<img src="cid:x">',
            attachments=[_att("log.txt", mime="text/plain", data=b"log"), _att("x.png", cid="x")],
        ),
    )
    assert msg.get_content_type() == "multipart/mixed"
    alt, attached = msg.get_payload()
    related = alt.get_payload()[1]
    assert related.get_content_type() == "multipart/related"
    assert related.get_payload()[1]["Content-ID"] == "<x>"
    assert attached.get_filename() == "log.txt"


@pytest.mark.parametrize("raw, expected", [
    ("reverie1", "reverie1"),
    ("<reverie1>", "reverie1"),
    ("cid:reverie1@orion", "reverie1@orion"),
    ("  ", None),
])
def test_content_id_is_normalized(raw, expected):
    assert _att("a.png", cid=raw).content_id == expected


@pytest.mark.parametrize("bad", ["a b", "a>b<c", "x\r\nBcc: evil@example.com", "a" * 201])
def test_invalid_content_id_is_rejected_at_the_boundary(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _att("a.png", cid=bad)


def test_duplicate_content_ids_are_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _req(body_html="<p/>", attachments=[_att("a.png", cid="x"), _att("b.png", cid="x")])
