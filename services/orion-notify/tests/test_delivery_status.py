"""The persisted notification status must say what actually happened.

Before 2026-08-30 `notify_requests.status` was hardcoded `"pending"` at every
call site and nothing anywhere updated it: 10,900 rows since 2026-07-24, 100%
pending, including emails Juniper confirmed receiving. A column that always
says pending is worse than no column -- a reader gets a wrong answer instead of
no answer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SERVICE_DIR = str(Path(__file__).resolve().parents[1])
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("SERVICE_NAME", "orion-notify")
os.environ.setdefault("SERVICE_VERSION", "0.1.0")
os.environ.setdefault("NODE_NAME", "athena")

from app.email_delivery import EmailOutcome, maybe_send_email  # noqa: E402


class _Payload:
    def __init__(self, severity="critical", **kw):
        self.severity = severity
        self.notification_id = "n1"
        self.event_kind = "test.kind"
        self.channels_requested = kw.get("channels", ["email"])
        self.recipient_group = kw.get("recipient_group", "juniper_primary")
        for k, v in kw.items():
            setattr(self, k, v)


class _Transport:
    def __init__(self, exc=None):
        self.exc = exc
        self.sent = []

    def send(self, payload):
        if self.exc:
            raise self.exc
        self.sent.append(payload)


# --------------------------------------------------------------------------
# the outcome is returned, not discarded
# --------------------------------------------------------------------------

def test_no_transport_is_skipped_not_pending() -> None:
    out = maybe_send_email(None, _Payload())
    assert out.status == "skipped"
    assert out.reason == "smtp_transport_unconfigured"


def test_a_successful_send_reports_sent(monkeypatch) -> None:
    import app.email_delivery as ed

    monkeypatch.setattr(ed, "should_send_email", lambda p: (True, "policy_ok"))
    t = _Transport()
    out = maybe_send_email(t, _Payload())
    assert out.status == "sent"
    assert len(t.sent) == 1


def test_a_raising_transport_reports_failed_with_the_error(monkeypatch) -> None:
    import app.email_delivery as ed

    monkeypatch.setattr(ed, "should_send_email", lambda p: (True, "policy_ok"))
    out = maybe_send_email(_Transport(exc=RuntimeError("smtp auth rejected")), _Payload())
    assert out.status == "failed"
    assert "RuntimeError" in out.reason and "smtp auth rejected" in out.reason


def test_a_failed_send_is_never_reported_as_sent(monkeypatch) -> None:
    """The whole point: a failure must not read as success. This is the shape
    that let 663 gateway timeouts get stored as `result_status=ok`."""
    import app.email_delivery as ed

    monkeypatch.setattr(ed, "should_send_email", lambda p: (True, "ok"))
    for exc in (RuntimeError("x"), OSError("y"), ValueError("z"), TimeoutError("t")):
        assert maybe_send_email(_Transport(exc=exc), _Payload()).status == "failed"


def test_policy_decline_is_skipped_and_carries_the_reason(monkeypatch) -> None:
    import app.email_delivery as ed

    monkeypatch.setattr(ed, "should_send_email", lambda p: (False, "severity_below_threshold"))
    out = maybe_send_email(_Transport(), _Payload())
    assert out.status == "skipped"
    assert out.reason == "severity_below_threshold"


def test_immediate_critical_only_defers_rather_than_declining(monkeypatch) -> None:
    """"deferred", not "skipped": attention_escalation.py emails exactly
    severity=="error" attentions past their ack deadline, so an email may still
    follow. Live 2026-08-30, 37 of 46 error attentions escalated and every other
    severity escalated zero times -- calling those "no email" would make the
    column confidently wrong about the only class that reliably emails."""
    import app.email_delivery as ed

    monkeypatch.setattr(ed, "should_send_email", lambda p: (True, "ok"))
    t = _Transport()
    out = maybe_send_email(t, _Payload(severity="error"), immediate_critical_only=True)
    assert out.status == "deferred"
    assert "immediate_critical_only" in out.reason
    assert t.sent == [], "a deferred send must not touch the transport"


def test_maybe_send_email_always_returns_an_outcome(monkeypatch) -> None:
    """It used to return None on four of five paths; a None here silently maps
    back to "pending" and reintroduces the bug."""
    import app.email_delivery as ed

    for should in (True, False):
        monkeypatch.setattr(ed, "should_send_email", lambda p, _s=should: (_s, "r"))
        for transport in (None, _Transport(), _Transport(exc=RuntimeError("x"))):
            for critical_only in (True, False):
                out = maybe_send_email(
                    transport, _Payload(), immediate_critical_only=critical_only
                )
                assert isinstance(out, EmailOutcome), "every path must return an outcome"
                assert out.status in {"sent", "failed", "skipped"}



# --------------------------------------------------------------------------
# the mapping onto the persisted column
# --------------------------------------------------------------------------

def _request_status(outcome):
    from app.main import _request_status as fn
    return fn(outcome)


@pytest.mark.parametrize(
    "email_status,expected",
    [("sent", "sent"), ("failed", "failed"), ("skipped", "no_email")],
)
def test_outcome_maps_onto_the_persisted_status(email_status, expected) -> None:
    pytest.importorskip("fastapi")
    assert _request_status(EmailOutcome(email_status, "r")) == expected


def test_an_unknown_outcome_falls_back_to_pending() -> None:
    """"pending" now means what it says -- nothing attempted, nothing known --
    rather than being the only value the column ever held."""
    pytest.importorskip("fastapi")
    assert _request_status(None) == "pending"
    assert _request_status(EmailOutcome("weird", "r")) == "pending"


# --------------------------------------------------------------------------
# the endpoints themselves -- what actually gets persisted
#
# The previous guard here was a source-text grep. It counted characters, not
# values, so `status=_request_status(email_outcome) and "pending"` -- a FULL
# revert of this commit at both endpoints -- passed all 31 tests. These drive
# the real handlers and assert the status on the NotificationRecord that is
# actually published for persistence.
# --------------------------------------------------------------------------

def _app_request(monkeypatch, transport):
    """A fake Request whose app.state carries what the handlers read."""
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from app import main as m

    published: list = []
    coros: list = []

    async def _capture(bus, channel, record):
        published.append(record)

    monkeypatch.setattr(m, "_publish_persistence_event", _capture)
    monkeypatch.setattr(m.asyncio, "create_task", lambda c: coros.append(c) or None)
    monkeypatch.setattr(m, "_check_token", lambda *_a, **_k: None)

    state = SimpleNamespace(
        bus=object(),
        email_transport=transport,
        policy=SimpleNamespace(
            evaluate=lambda payload, now: SimpleNamespace(
                ack_deadline_minutes=60, escalation_channels=["email"],
                should_send_email=True, drop_reason=None, action="publish",
            )
        ),
    )
    return m, SimpleNamespace(app=SimpleNamespace(state=state)), published, coros


async def _drain(coros, published):
    for c in coros:
        try:
            await c
        except Exception:
            pass
    return published


@pytest.mark.asyncio
async def test_notify_endpoint_persists_sent_when_the_email_goes_out(monkeypatch) -> None:
    m, req, published, coros = _app_request(monkeypatch, _Transport())
    monkeypatch.setattr(m, "should_send_email", lambda p: (True, "policy_ok"))
    from orion.schemas.notify import NotificationRequest

    await m.notify(
        payload=NotificationRequest(
            source_service="test", event_kind="test.kind", severity="critical",
            title="t", body_text="b",
        ),
        request=req,
    )
    records = await _drain(coros, published)
    assert records, "no NotificationRecord was published"
    assert records[0].status == "sent"


@pytest.mark.asyncio
async def test_notify_endpoint_persists_failed_when_smtp_raises(monkeypatch) -> None:
    """The load-bearing one: a failure must never persist as success. This is
    the shape that let 663 gateway timeouts get stored as result_status=ok."""
    m, req, published, coros = _app_request(monkeypatch, _Transport(exc=RuntimeError("smtp down")))
    monkeypatch.setattr(m, "should_send_email", lambda p: (True, "policy_ok"))
    from orion.schemas.notify import NotificationRequest

    await m.notify(
        payload=NotificationRequest(
            source_service="test", event_kind="test.kind", severity="critical",
            title="t", body_text="b",
        ),
        request=req,
    )
    records = await _drain(coros, published)
    assert records[0].status == "failed"
    assert records[0].status != "pending", "the original bug, restored"


@pytest.mark.asyncio
async def test_notify_endpoint_persists_no_email_when_policy_declines(monkeypatch) -> None:
    m, req, published, coros = _app_request(monkeypatch, _Transport())
    monkeypatch.setattr(m, "should_send_email", lambda p: (False, "severity_below_threshold"))
    from orion.schemas.notify import NotificationRequest

    await m.notify(
        payload=NotificationRequest(
            source_service="test", event_kind="test.kind", severity="info",
            title="t", body_text="b",
        ),
        request=req,
    )
    records = await _drain(coros, published)
    assert records[0].status == "no_email"
    assert records[0].drop_reason == "severity_below_threshold", "the WHY must survive"


@pytest.mark.asyncio
async def test_no_endpoint_can_persist_a_hardcoded_pending(monkeypatch) -> None:
    """Behavioural replacement for the source grep that a full revert walked
    through. Every email outcome must reach the record."""
    from orion.schemas.notify import NotificationRequest

    cases = [
        (_Transport(), (True, "ok"), "sent"),
        (_Transport(exc=RuntimeError("x")), (True, "ok"), "failed"),
        (None, (True, "ok"), "no_email"),
        (_Transport(), (False, "declined"), "no_email"),
    ]
    for transport, policy, expected in cases:
        m, req, published, coros = _app_request(monkeypatch, transport)
        monkeypatch.setattr(m, "should_send_email", lambda p, _r=policy: _r)
        await m.notify(
            payload=NotificationRequest(
                source_service="test", event_kind="test.kind", severity="critical",
                title="t", body_text="b",
            ),
            request=req,
        )
        records = await _drain(coros, published)
        assert records[0].status == expected, f"{transport}/{policy} -> {records[0].status}"


# --------------------------------------------------------------------------
# the logger must actually reach stdout
# --------------------------------------------------------------------------

def test_importing_the_service_configures_a_root_handler() -> None:
    """Verified live in the running container on 2026-08-30, BEFORE this fix:
    effective level WARNING, `logging.getLogger().handlers == []`, INFO records
    dropped. uvicorn configures its own loggers, so `docker logs` showed 203
    access lines in 24h and zero application lines while 230 notifications were
    created. Every [NOTIFY] breadcrumb this service has ever written went
    nowhere -- which is why nobody noticed delivery accounting did not exist.
    """
    pytest.importorskip("fastapi")
    import logging

    import app.main  # noqa: F401  (import configures logging as a side effect)

    root = logging.getLogger()
    assert root.handlers, "no root handler: every [NOTIFY] line is dropped"
    assert logging.getLogger("orion-notify").getEffectiveLevel() <= logging.INFO, (
        "INFO is the level every [NOTIFY] breadcrumb is emitted at"
    )


def test_the_root_handler_writes_to_stdout() -> None:
    """Asserts the handler's stream, not captured output: basicConfig runs at
    import and binds the real sys.stdout before pytest swaps it, so capsys
    cannot see the line. Verified end to end instead by emitting inside the
    running container, which printed the record to stdout.

    stdout specifically, because that is what `docker logs` shows and what was
    empty of application lines for the life of this service.
    """
    pytest.importorskip("fastapi")
    import logging

    import app.main  # noqa: F401

    streams = [
        getattr(h, "stream", None)
        for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler)
    ]
    assert streams, "no StreamHandler on the root logger"
    assert any(getattr(s, "name", None) == "<stdout>" or s is sys.stdout for s in streams), (
        f"root handler does not write to stdout: {streams}"
    )


def test_a_partial_recipient_refusal_is_not_reported_as_sent(monkeypatch) -> None:
    """smtplib.send_message raises only when EVERY recipient is refused; on a
    partial refusal it RETURNS the refused addresses and does not raise. The
    return was discarded, so a partially-refused message reported a clean send.
    NOTIFY_EMAIL_TO is comma-separated, so multi-recipient is supported config.
    """
    import smtplib

    import app.email_delivery as ed

    class PartiallyRefusing:
        def send(self, payload):
            raise smtplib.SMTPRecipientsRefused({"nope@example.com": (550, b"No such user")})

    monkeypatch.setattr(ed, "should_send_email", lambda p: (True, "ok"))
    out = maybe_send_email(PartiallyRefusing(), _Payload())
    assert out.status == "failed"
    assert "Refused" in out.reason or "refused" in out.reason.lower()


def test_deferred_maps_to_pending_not_no_email() -> None:
    """The 37-of-46 case. `/attention/request` skips immediate email for
    severity="error", but attention_escalation.py emails exactly that class past
    the ack deadline -- so "no_email" would be confidently wrong about the only
    severity that reliably emails, while "pending" is honest."""
    pytest.importorskip("fastapi")
    assert _request_status(EmailOutcome("deferred", "immediate_critical_only_severity_error")) == "pending"
    assert _request_status(EmailOutcome("deferred", "x")) != "no_email"


def test_an_unmapped_outcome_is_logged_loudly(caplog) -> None:
    """The fallback returns the exact known-bad value this change exists to
    remove, so it must be visible. A silent fallback reproduces 100%-pending and
    looks identical to the pre-fix state."""
    pytest.importorskip("fastapi")
    import logging

    with caplog.at_level(logging.WARNING, logger="orion-notify"):
        assert _request_status(EmailOutcome("brand_new_status", "r")) == "pending"
    assert any("unmapped_email_outcome" in r.getMessage() for r in caplog.records), (
        "the degraded fallback must be logged"
    )


@pytest.mark.asyncio
async def test_escalation_captures_the_email_outcome() -> None:
    """The third call site. It sends the REAL escalation email to Juniper and
    discarded its outcome -- the exact bug this commit exists to fix, still live
    at a site the first pass missed."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.attention_escalation import run_attention_escalation_once

    calls = []

    class Recording:
        def send(self, payload):
            calls.append(payload)

    old = datetime.now(timezone.utc) - timedelta(minutes=90)
    row = {
        "attention_id": "att-outcome", "severity": "error", "require_ack": True,
        "acked_at": None, "escalated_at": None, "created_at": old.isoformat(),
        "ack_deadline_minutes": 60, "reason": "attention_request",
        "message": "m", "source_service": "svc",
        "context": {"escalation_channels": ["email"]},
    }
    count = await run_attention_escalation_once(
        email_transport=Recording(),
        policy=SimpleNamespace(evaluate=lambda p, n: SimpleNamespace(
            ack_deadline_minutes=60, escalation_channels=["email"])),
        proxy_get=AsyncMock(return_value=[row]),
        proxy_post=AsyncMock(return_value={"status": "escalated"}),
        hub_url_base="",
    )
    assert count == 1
    assert len(calls) == 1, "the escalation email must actually be attempted"

    import app.attention_escalation as ae
    import inspect
    src = inspect.getsource(ae.run_attention_escalation_once)
    assert "outcome = maybe_send_email(" in src, (
        "the escalation call site must bind the outcome, not discard it"
    )
    assert "email_status=%s" in src, "and must report it in the log line"


def test_the_real_transport_treats_a_partial_refusal_as_a_failure(monkeypatch) -> None:
    """Exercises orion/notify/transport.py itself, not a stub that raises.

    The earlier partial-refusal test used a fake transport that raised directly,
    so it never touched the code that discards `send_message`'s return -- the
    actual defect. smtplib returns refused recipients instead of raising when
    only SOME are refused.
    """
    import smtplib

    from orion.notify.transport import EmailTransport

    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, msg):
            return {"nope@example.com": (550, b"No such user")}

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    tx = EmailTransport(
        smtp_host="h", smtp_port=587, smtp_username="u", smtp_password="p",
        use_tls=True, default_from="a@b.c",
        default_to=["ok@example.com", "nope@example.com"],
    )
    with pytest.raises(smtplib.SMTPRecipientsRefused):
        tx.send(_Payload(title="t", body_text="b", body_md=None, attachments=[]))
