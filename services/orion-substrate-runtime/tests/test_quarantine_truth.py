"""Tests for durable reducer quarantine truth semantics and operator acknowledgement."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.grammar_truth import build_substrate_grammar_truth
from app.quarantine_ack import clear_quarantine_acks_for_tests, quarantine_ack_snapshot
from app.reducer_health import clear_health_for_tests
from orion.substrate.transport_loop.constants import TRANSPORT_GRAMMAR_CURSOR_NAME


@pytest.fixture(autouse=True)
def _clear_state() -> None:
    clear_health_for_tests()
    clear_quarantine_acks_for_tests()


def _mock_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.grammar_truth.get_settings",
        lambda: MagicMock(
            orion_bus_enabled=True,
            enable_execution_trajectory_reducer=True,
            enable_transport_bus_reducer=True,
            enable_biometrics_node_reducer=True,
            enable_biometrics_pressure_organ=True,
            enable_node_pressure_reducer=True,
            substrate_cursor_lag_resync_hours=6.0,
            reducer_heartbeat_stale_sec=120.0,
            grammar_poll_interval_sec=5.0,
            substrate_cursor_tail_seed_on_lag=False,
            substrate_cursor_reset_operator_token="tok",
            biometrics_grammar_batch_limit=50,
            execution_grammar_batch_limit=100,
            transport_grammar_batch_limit=500,
            accepted_pressure_grammar_channel="orion:grammar:accepted-pressure",
            grammar_event_channel="orion:grammar:event",
            publish_accepted_pressure_grammar=True,
        ),
    )
    monkeypatch.setattr("app.grammar_truth.has_recent_tail_seed", lambda: False)
    monkeypatch.setattr("app.grammar_truth.has_cold_start_tail_seed", lambda: False)
    monkeypatch.setattr("app.grammar_truth.last_reset_skipped_history", lambda: False)
    monkeypatch.setattr(
        "app.grammar_truth.tail_seed_snapshot",
        lambda: {"count": 0, "latest": None, "recent": []},
    )
    monkeypatch.setattr(
        "app.grammar_truth.cursor_reset_snapshot",
        lambda: {"count": 0, "last": None, "recent": []},
    )


def _base_store_mock() -> MagicMock:
    store = MagicMock()
    store.cursor_positions.return_value = []
    store.grammar_cursor_metrics.side_effect = lambda name: {
        "cursor_name": name,
        "pending_backlog": 0,
        "stream_lag_sec": 0,
        "cursor_wall_lag_sec": 0,
        "head_event_created_at": None,
        "head_event_id": None,
    }
    return store


def test_truth_degrades_on_unacknowledged_quarantine(monkeypatch) -> None:
    _mock_settings(monkeypatch)
    store = _base_store_mock()
    store.quarantine_summary.return_value = {
        "unacknowledged_quarantine_count_by_reducer": {"transport_bus": 1},
        "unacknowledged_quarantine_count_by_cursor": {
            TRANSPORT_GRAMMAR_CURSOR_NAME: 1,
        },
        "quarantine_by_reducer": {
            "transport_bus": {
                "unacknowledged_count": 1,
                "recent_examples": [
                    {
                        "event_id": "gev_poison",
                        "trace_id": "bus.transport:poison01",
                        "reason": "poison payload",
                        "quarantined_at": "2026-06-15T12:00:00+00:00",
                    }
                ],
            }
        },
    }

    payload = build_substrate_grammar_truth(store)
    assert payload["degraded"] is True
    assert f"reducer_quarantine_present:{TRANSPORT_GRAMMAR_CURSOR_NAME}" in payload[
        "degraded_reasons"
    ]
    assert payload["unacknowledged_quarantine_count_by_reducer"]["transport_bus"] == 1
    assert payload["quarantine_by_reducer"]["transport_bus"]["recent_examples"][0][
        "event_id"
    ] == "gev_poison"
    health = payload["reducer_health_by_name"]["transport_bus"]
    assert health["unacknowledged_quarantine_count"] == 1


def test_truth_reports_quarantine_after_health_snapshot_reset(monkeypatch) -> None:
    _mock_settings(monkeypatch)
    store = _base_store_mock()
    store.quarantine_summary.return_value = {
        "unacknowledged_quarantine_count_by_reducer": {"transport_bus": 1},
        "unacknowledged_quarantine_count_by_cursor": {
            TRANSPORT_GRAMMAR_CURSOR_NAME: 1,
        },
        "quarantine_by_reducer": {
            "transport_bus": {
                "unacknowledged_count": 1,
                "recent_examples": [{"event_id": "gev_poison", "trace_id": None, "reason": "x", "quarantined_at": "2026-06-15T12:00:00+00:00"}],
            }
        },
    }
    clear_health_for_tests()
    payload = build_substrate_grammar_truth(store)
    assert payload["degraded"] is True
    assert f"reducer_quarantine_present:{TRANSPORT_GRAMMAR_CURSOR_NAME}" in payload[
        "degraded_reasons"
    ]


def test_truth_healthy_when_quarantine_acknowledged(monkeypatch) -> None:
    _mock_settings(monkeypatch)
    store = _base_store_mock()
    store.quarantine_summary.return_value = {
        "unacknowledged_quarantine_count_by_reducer": {},
        "unacknowledged_quarantine_count_by_cursor": {},
        "quarantine_by_reducer": {},
    }

    from app.reducer_health import record_tick

    for reducer_key, cursor_name in (
        ("biometrics", "biometrics_grammar_consumer"),
        ("execution_trajectory", "execution_grammar_reducer"),
        ("transport_bus", "transport_grammar_reducer"),
    ):
        record_tick(reducer_key, cursor_name=cursor_name, enabled=True)

    payload = build_substrate_grammar_truth(store)
    assert payload["ok"] is True
    assert "reducer_quarantine_present:" not in " ".join(payload["degraded_reasons"])


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN", "test-operator-token")
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused:5432/unused")

    import app.settings as settings_mod

    settings_mod._settings = None
    import app.main as main_mod

    main_mod.worker._store = MagicMock()
    return TestClient(main_mod.app)


def test_quarantine_ack_requires_operator_token(client: TestClient) -> None:
    resp = client.post(
        f"/grammar/quarantine/ack?cursor_name={TRANSPORT_GRAMMAR_CURSOR_NAME}&event_id=gev_x"
    )
    assert resp.status_code == 401


def test_quarantine_ack_clears_degraded_reason(client: TestClient, monkeypatch) -> None:
    import app.main as main_mod

    main_mod.worker._store.acknowledge_quarantine.return_value = 1
    resp = client.post(
        f"/grammar/quarantine/ack?cursor_name={TRANSPORT_GRAMMAR_CURSOR_NAME}&event_id=gev_x",
        headers={"X-Orion-Operator-Token": "test-operator-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["acknowledged_count"] == 1
    snap = quarantine_ack_snapshot()
    assert snap["count"] == 1
    assert snap["last"]["event_id"] == "gev_x"


def test_quarantine_ack_all_for_cursor(client: TestClient) -> None:
    import app.main as main_mod

    main_mod.worker._store.acknowledge_quarantine.return_value = 3
    resp = client.post(
        f"/grammar/quarantine/ack?cursor_name={TRANSPORT_GRAMMAR_CURSOR_NAME}&ack_all=true",
        headers={"X-Orion-Operator-Token": "test-operator-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["ack_all"] is True
    assert resp.json()["acknowledged_count"] == 3
