"""Issue #3409: email to anyone but Juniper must not quote a core-page or lived-memory sentence.
Made-up text only."""
from __future__ import annotations

import smtplib

import pytest

from orion.memory.mesh_edge import MeshEdgeGuard, MeshEdgeLeak, MeshEdgeUnavailable
from orion.notify.transport import EmailTransport
from orion.schemas.notify import NotificationRequest

SENTENCE = "Robin told me the blue kite got stuck in the old oak tree."
JUNIPER = "juniper@example.test"


class _FakeSMTP:
    sent: list = []

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def starttls(self):
        pass

    def login(self, *_a):
        pass

    def send_message(self, msg):
        _FakeSMTP.sent.append(msg["To"])
        return {}


@pytest.fixture(autouse=True)
def _smtp(monkeypatch):
    _FakeSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)


def _transport(guard_loader):
    return EmailTransport("smtp.test", 25, "", "", False, "orion@example.test", [JUNIPER], guard_loader=guard_loader)


def _req(body: str) -> NotificationRequest:
    return NotificationRequest(source_service="test", event_kind="test", severity="info", title="t", body_text=body)


def test_quoting_a_page_sentence_to_someone_else_is_refused():
    guard = MeshEdgeGuard.from_texts([("page:juniper", SENTENCE)])
    with pytest.raises(MeshEdgeLeak):
        _transport(lambda: guard).send(_req(f"FYI: {SENTENCE}"), to_override=["friend@example.test"])
    assert _FakeSMTP.sent == []


def test_juniper_gets_her_own_memories_without_a_check():
    def never():
        raise AssertionError("no check for Juniper-only email")

    _transport(never).send(_req(f"FYI: {SENTENCE}"))
    assert _FakeSMTP.sent == [JUNIPER]


def test_clean_mail_to_someone_else_is_sent_and_an_unloadable_guard_refuses():
    guard = MeshEdgeGuard.from_texts([("page:juniper", SENTENCE)])
    _transport(lambda: guard).send(_req("The build is green."), to_override=["friend@example.test"])
    assert _FakeSMTP.sent == ["friend@example.test"]

    def down():
        raise MeshEdgeUnavailable("down")

    with pytest.raises(MeshEdgeUnavailable):
        _transport(down).send(_req("The build is green."), to_override=["friend@example.test"])
