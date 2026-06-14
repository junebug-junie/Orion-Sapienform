"""Tests for protected substrate cursor reset endpoint."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.cursor_reset import clear_cursor_resets_for_tests, cursor_reset_snapshot
from orion.substrate.biometrics_loop.constants import GRAMMAR_CURSOR_NAME


@pytest.fixture(autouse=True)
def _clear_resets() -> None:
    clear_cursor_resets_for_tests()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN", "test-operator-token")
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused:5432/unused")

    import app.settings as settings_mod

    settings_mod._settings = None
    import app.main as main_mod

    main_mod.worker._store = MagicMock()
    return TestClient(main_mod.app)


def test_reset_requires_operator_token(client: TestClient) -> None:
    resp = client.post(
        f"/grammar/cursor/reset?cursor_name={GRAMMAR_CURSOR_NAME}&mode=earliest"
    )
    assert resp.status_code == 401


def test_reset_rejects_invalid_cursor_name(client: TestClient) -> None:
    resp = client.post(
        "/grammar/cursor/reset?cursor_name=not-a-real-cursor&mode=earliest",
        headers={"X-Orion-Operator-Token": "test-operator-token"},
    )
    assert resp.status_code == 400
    assert "invalid cursor_name" in resp.json()["detail"]


def test_reset_rejects_invalid_mode(client: TestClient) -> None:
    resp = client.post(
        f"/grammar/cursor/reset?cursor_name={GRAMMAR_CURSOR_NAME}&mode=jump",
        headers={"X-Orion-Operator-Token": "test-operator-token"},
    )
    assert resp.status_code == 400
    assert "invalid mode" in resp.json()["detail"]


def test_reset_timestamp_requires_timezone(client: TestClient) -> None:
    resp = client.post(
        f"/grammar/cursor/reset?cursor_name={GRAMMAR_CURSOR_NAME}&mode=timestamp&at=2026-06-01T00:00:00",
        headers={"X-Orion-Operator-Token": "test-operator-token"},
    )
    assert resp.status_code == 400
    assert "timezone-aware" in resp.json()["detail"]


def test_successful_reset_records_audit(client: TestClient) -> None:
    client.app.state  # ensure app loaded
    import app.main as main_mod

    main_mod.worker._store.reset_grammar_cursor.return_value = {
        "cursor_name": GRAMMAR_CURSOR_NAME,
        "mode": "earliest",
        "prior_created_at": "2026-06-01T00:00:00+00:00",
        "prior_event_id": "evt-old",
        "new_created_at": "1970-01-01T00:00:00+00:00",
        "new_event_id": "",
        "history_may_be_skipped": False,
    }
    resp = client.post(
        f"/grammar/cursor/reset?cursor_name={GRAMMAR_CURSOR_NAME}&mode=earliest",
        headers={"X-Orion-Operator-Token": "test-operator-token"},
    )
    assert resp.status_code == 200
    snap = cursor_reset_snapshot()
    assert snap["count"] == 1
    assert snap["last"]["cursor_name"] == GRAMMAR_CURSOR_NAME
    assert snap["last"]["mode"] == "earliest"
    assert snap["last"]["prior_event_id"] == "evt-old"


def test_tail_reset_marks_history_skipped_in_audit(client: TestClient) -> None:
    import app.main as main_mod

    main_mod.worker._store.reset_grammar_cursor.return_value = {
        "cursor_name": GRAMMAR_CURSOR_NAME,
        "mode": "tail",
        "prior_created_at": None,
        "prior_event_id": None,
        "new_created_at": datetime.now(timezone.utc).isoformat(),
        "new_event_id": "evt-tail",
        "history_may_be_skipped": True,
    }
    resp = client.post(
        f"/grammar/cursor/reset?cursor_name={GRAMMAR_CURSOR_NAME}&mode=tail",
        headers={"X-Orion-Operator-Token": "test-operator-token"},
    )
    assert resp.status_code == 200
    assert cursor_reset_snapshot()["last"]["history_may_be_skipped"] is True
