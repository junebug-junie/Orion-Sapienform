from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_notify import router

ATTN_ID = "8c0c1a20-9a9f-45b1-9aa0-1af6d1f7a6f3"


def _make_row(**overrides):
    base = dict(
        attention_id=ATTN_ID,
        notification_id=None,
        created_at=datetime(2026, 7, 1, 12, 0, 0),
        source_service="orion-cortex",
        severity="info",
        title="I want to talk",
        body_text="Can you check the latest summary?",
        context={},
        tags=[],
        correlation_id=None,
        session_id=None,
        attention_expires_at=None,
        attention_require_ack=True,
        attention_ack_deadline_minutes=None,
        attention_acked_at=None,
        attention_ack_type=None,
        attention_ack_actor=None,
        attention_ack_note=None,
        attention_escalated_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(router, prefix="/api/notify-read")

    fake_db = MagicMock()
    fake_row = _make_row()

    fake_query = MagicMock()
    fake_query.filter.return_value.first.return_value = fake_row
    fake_db.query.return_value = fake_query

    monkeypatch.setattr("app.api_notify.get_session", lambda: fake_db)
    monkeypatch.setattr("app.api_notify.remove_session", lambda: None)

    with TestClient(app) as test_client:
        yield test_client, fake_db, fake_row


def test_attention_ack_persists_fields_on_notify_requests_row(client):
    """Regression test: the ack must actually set attention_acked_at/ack_type/
    ack_actor/ack_note on the NotificationRequestDB row (the bug was that the
    old code published a NotificationReceiptEvent keyed on message_id, which
    never touched these columns at all)."""
    test_client, fake_db, fake_row = client

    resp = test_client.post(
        f"/api/notify-read/attention/{ATTN_ID}/ack",
        json={
            "attention_id": ATTN_ID,
            "ack_type": "dismissed",
            "actor": "juniper",
            "note": "handled",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "acked"
    assert body["ack_type"] == "dismissed"
    assert body["ack_actor"] == "juniper"
    assert body["ack_note"] == "handled"

    # The underlying row was actually mutated (this is the persistence check).
    assert fake_row.attention_acked_at is not None
    assert fake_row.attention_ack_type == "dismissed"
    assert fake_row.attention_ack_actor == "juniper"
    assert fake_row.attention_ack_note == "handled"
    fake_db.commit.assert_called_once()
    fake_db.refresh.assert_called_once()


def test_attention_ack_idempotent_does_not_overwrite(client):
    test_client, fake_db, fake_row = client
    fake_row.attention_acked_at = datetime(2026, 6, 19, 12, 0, 0)
    fake_row.attention_ack_type = "seen"
    fake_row.attention_ack_actor = "juniper"
    fake_row.attention_ack_note = None

    resp = test_client.post(
        f"/api/notify-read/attention/{ATTN_ID}/ack",
        json={"attention_id": ATTN_ID, "ack_type": "dismissed", "actor": "someone_else"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ack_type"] == "seen"  # unchanged, first ack wins
    fake_db.commit.assert_not_called()


def test_attention_ack_missing_row_returns_404_not_crash(client):
    test_client, fake_db, _fake_row = client
    fake_db.query.return_value.filter.return_value.first.return_value = None

    resp = test_client.post(
        f"/api/notify-read/attention/{ATTN_ID}/ack",
        json={"attention_id": ATTN_ID, "ack_type": "seen"},
    )

    assert resp.status_code == 404
    fake_db.commit.assert_not_called()
