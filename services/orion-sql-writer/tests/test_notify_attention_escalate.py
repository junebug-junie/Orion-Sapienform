from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api_notify import router


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(router, prefix="/api/notify-read")

    fake_db = MagicMock()
    fake_row = MagicMock()
    fake_row.attention_id = "att-123"
    fake_row.attention_escalated_at = None

    fake_query = MagicMock()
    fake_query.filter.return_value.first.return_value = fake_row
    fake_db.query.return_value = fake_query

    monkeypatch.setattr("app.api_notify.get_session", lambda: fake_db)
    monkeypatch.setattr("app.api_notify.remove_session", lambda: None)

    with TestClient(app) as test_client:
        yield test_client, fake_db, fake_row


def test_mark_attention_escalated_sets_timestamp(client):
    test_client, fake_db, fake_row = client

    resp = test_client.post("/api/notify-read/attention/att-123/escalate")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "escalated"
    assert body["attention_id"] == "att-123"
    assert fake_row.attention_escalated_at is not None
    fake_db.commit.assert_called_once()


def test_mark_attention_escalated_idempotent(client):
    test_client, _fake_db, fake_row = client
    fake_row.attention_escalated_at = datetime(2026, 6, 19, 12, 0, 0)

    resp = test_client.post("/api/notify-read/attention/att-123/escalate")

    assert resp.status_code == 200
    assert resp.json()["status"] == "already_escalated"
