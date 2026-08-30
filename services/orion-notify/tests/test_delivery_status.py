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


def test_immediate_critical_only_skips_non_critical(monkeypatch) -> None:
    import app.email_delivery as ed

    monkeypatch.setattr(ed, "should_send_email", lambda p: (True, "ok"))
    t = _Transport()
    out = maybe_send_email(t, _Payload(severity="warning"), immediate_critical_only=True)
    assert out.status == "skipped"
    assert "immediate_critical_only" in out.reason
    assert t.sent == [], "a skipped send must not touch the transport"


def test_maybe_send_email_always_returns_an_outcome(monkeypatch) -> None:
    """It used to return None on four of five paths; a None here silently maps
    back to "pending" and reintroduces the bug."""
    import app.email_delivery as ed
    import inspect

    for should in (True, False):
        monkeypatch.setattr(ed, "should_send_email", lambda p, _s=should: (_s, "r"))
        for transport in (None, _Transport(), _Transport(exc=RuntimeError("x"))):
            for critical_only in (True, False):
                out = maybe_send_email(
                    transport, _Payload(), immediate_critical_only=critical_only
                )
                assert isinstance(out, EmailOutcome), "every path must return an outcome"
                assert out.status in {"sent", "failed", "skipped"}

    assert "None" not in str(inspect.signature(maybe_send_email).return_annotation)


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


def test_no_call_site_hardcodes_pending_on_a_persisted_record() -> None:
    """Guards the actual regression: a future edit reintroducing a literal
    `status="pending"` on a NotificationRecord. The one remaining literal is the
    attention record, whose status is DERIVED at read time from
    attention_acked_at (api_notify.py::_attention_to_schema), not stored."""
    src = Path(SERVICE_DIR, "app", "main.py").read_text()
    assert src.count('status="pending"') <= 1, (
        "a persisted-record call site hardcodes pending again"
    )
    assert src.count("_request_status(email_outcome)") == 2, (
        "both email-bearing endpoints must record the real outcome"
    )


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
