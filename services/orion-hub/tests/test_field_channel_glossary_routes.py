from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import field_channel_glossary_routes  # noqa: E402


def _field_state_row(overrides: dict) -> dict:
    node_vectors = {
        "node:athena": {
            "availability": 1.0,
            "cpu_pressure": 0.0,
            "expected_offline_suppression": 1.0,
            "bus_health": 1.0,
            "delivery_confidence": 1.0,
        }
    }
    node_vectors["node:athena"].update(overrides)
    return {
        "schema_version": "field.state.v1",
        "generated_at": "2026-07-20T00:00:00+00:00",
        "tick_id": "tick-test",
        "node_vectors": node_vectors,
        "capability_vectors": {},
        "edges": [],
        "recent_perturbations": [],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    field_channel_glossary_routes._engine_instance = None
    app = FastAPI()
    app.include_router(field_channel_glossary_routes.router)
    return TestClient(app)


def _fake_engine_with_rows(field_jsons: list[dict]):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value = [(json.dumps(fj),) for fj in field_jsons]
    return fake_engine


def test_channels_endpoint_returns_35_entries(client):
    r = client.get("/api/field-channel-glossary/channels")
    assert r.status_code == 200
    body = r.json()
    assert len(body["channels"]) == 35
    assert len(body["categories"]) == 7


def test_health_endpoint_classifies_dead_channel(client):
    rows = [_field_state_row({"cpu_pressure": 0.0}) for _ in range(5)]
    fake_engine = _fake_engine_with_rows(rows)

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")

    assert r.status_code == 200
    body = r.json()
    assert body["window_hours"] == 1
    assert body["row_count"] == 5
    by_channel = {c["channel"]: c for c in body["channels"]}
    assert by_channel["cpu_pressure"]["verdict"] == "dead"
    assert by_channel["cpu_pressure"]["clean"] is False


def test_health_endpoint_classifies_live_channel(client):
    values = [0.1, 0.4, 0.15, 0.6, 0.2]
    rows = [_field_state_row({"cpu_pressure": v}) for v in values]
    fake_engine = _fake_engine_with_rows(rows)

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")

    body = r.json()
    by_channel = {c["channel"]: c for c in body["channels"]}
    assert by_channel["cpu_pressure"]["verdict"] == "live"
    assert by_channel["cpu_pressure"]["clean"] is True
    assert by_channel["cpu_pressure"]["sample_count"] == 5


def test_health_endpoint_defaults_invalid_hours_to_default(client):
    fake_engine = _fake_engine_with_rows([_field_state_row({})])

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=999")

    assert r.status_code == 200
    assert r.json()["window_hours"] == field_channel_glossary_routes.DEFAULT_HOURS


def test_build_channel_series_quiet_zero_channel_is_dead_not_never_produced():
    # transport_pressure is never in PRESSURE_CHANNELS/HIGHER_IS_BETTER_CHANNELS,
    # so a tick where every source reads exactly 0.0 means
    # collect_field_channel_pressures() drops it from the merge -- but the
    # raw node vector still genuinely carries the key at 0.0 (reconcile
    # seeds it). build_channel_series() must record that as a real 0.0
    # sample, not silently treat the channel as absent/never-wired.
    rows = [_field_state_row({"transport_pressure": 0.0}) for _ in range(3)]
    series, unparsable = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["transport_pressure"]
    )
    assert unparsable == 0
    assert series["transport_pressure"] == [0.0, 0.0, 0.0]
    assert field_channel_glossary_routes.classify_channel_series(series["transport_pressure"]) == "dead"


def test_build_channel_series_genuinely_never_produced_when_key_absent_everywhere():
    rows = [_field_state_row({}) for _ in range(3)]
    series, _unparsable = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["transport_pressure"]
    )
    assert series["transport_pressure"] == []
    assert field_channel_glossary_routes.classify_channel_series(series["transport_pressure"]) == "never_produced"


def test_build_channel_series_counts_unparsable_rows_separately_from_dead():
    rows = [_field_state_row({}), {"not": "a valid field state"}, _field_state_row({})]
    series, unparsable = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]
    )
    assert unparsable == 1
    assert len(series["cpu_pressure"]) == 2


def test_health_endpoint_reverses_desc_rows_back_to_chronological_order(client):
    # The route issues `ORDER BY generated_at DESC LIMIT :row_cap` -- when
    # Postgres has already truncated to ROW_CAP rows (simulated here by the
    # mock returning exactly that many, newest-first), the route must
    # reverse them back to ascending chronological order before building
    # series. Otherwise "last" (meant to be the most recent reading) would
    # actually be the oldest one in the truncated set, and a monotonically
    # climbing real signal would look like it was falling.
    chronological_values = [0.1, 0.4, 0.7]  # oldest -> newest
    rows_desc_from_db = [
        _field_state_row({"cpu_pressure": v}) for v in reversed(chronological_values)
    ]
    fake_engine = _fake_engine_with_rows(rows_desc_from_db)

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")

    by_channel = {c["channel"]: c for c in r.json()["channels"]}
    cpu = by_channel["cpu_pressure"]
    # If the DESC rows were used as-is (bug), "last" would read 0.1 (the
    # oldest tick, first in the DESC list) instead of 0.7 (the true most
    # recent reading).
    assert cpu["last"] == pytest.approx(0.7)
    assert cpu["min"] == pytest.approx(0.1)
    assert cpu["max"] == pytest.approx(0.7)
