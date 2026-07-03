"""HTTP tests for the self-observability summary route (self-observability v2)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
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

from scripts import hub_presence, substrate_observability_routes  # noqa: E402

_NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)

_SELF_STATE = {
    "attention_schema_type": "focused_single",
    "attention_dwell_ticks": 4,
    "attention_node_count": 1,
    "hub_presence": {"connection_health": "active"},
    "overall_condition": "steady",
    "summary_labels": ["field_active"],
    "generated_at": "2026-07-03T11:59:00+00:00",
}

_BROADCAST = {
    "selected_description": "unresolved contradiction in transport lane",
    "attended_node_ids": ["node:a"],
    "dwell_ticks": 4,
    "coalition_stability_score": 0.9,
    "generated_at": "2026-07-03T11:59:30+00:00",
}

_CANDIDATES = [
    {"signal_type": "curiosity_candidate", "signal_strength": 0.9, "evidence_summary": "gap a"},
    {"signal_type": "curiosity_candidate", "signal_strength": 0.4, "evidence_summary": "gap b"},
]

_PRESENCE_ROW = {"last_turn_age_sec": 12.0, "turns_per_minute": 1.2, "connection_health": "active"}


@pytest.fixture(autouse=True)
def _reset_presence():
    hub_presence.reset()
    yield
    hub_presence.reset()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_observability_routes.router)
    return TestClient(app)


def _fake_engine(rows_by_table: dict[str, dict | None]):
    """Engine whose execute() keys responses off the table name in the SQL."""
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        sql = str(stmt)
        m = MagicMock()
        result = None
        for table, row in rows_by_table.items():
            if table in sql:
                if isinstance(row, Exception):
                    raise row
                result = row
                break
        m.mappings.return_value.first.return_value = result
        return m

    conn.execute.side_effect = execute
    return fake_engine


def _all_rows() -> dict[str, dict | None]:
    return {
        "substrate_self_state": {"self_state_json": _SELF_STATE, "generated_at": _NOW},
        "substrate_attention_broadcast_projection": {"projection_json": _BROADCAST, "generated_at": _NOW},
        "substrate_endogenous_curiosity_candidates": {"candidates_json": _CANDIDATES, "generated_at": _NOW},
        "substrate_hub_presence": {"presence_json": _PRESENCE_ROW, "generated_at": _NOW},
    }


def test_summary_full_contract_shape(client):
    with patch.object(substrate_observability_routes, "_engine", return_value=_fake_engine(_all_rows())):
        r = client.get("/api/substrate/observability/summary")

    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"generated_at", "self_state", "attention_broadcast", "curiosity", "hub_presence"}
    assert body["self_state"]["attention_schema_type"] == "focused_single"
    assert body["self_state"]["overall_condition"] == "steady"
    assert body["attention_broadcast"]["dwell_ticks"] == 4
    assert body["attention_broadcast"]["coalition_stability_score"] == 0.9
    assert body["curiosity"]["gap_count"] == 2
    assert [s["evidence_summary"] for s in body["curiosity"]["signals"]] == ["gap a", "gap b"]
    assert body["hub_presence"]["connection_health"] == "active"
    assert body["hub_presence"]["generated_at"] == _NOW.isoformat()


def test_summary_each_section_degrades_to_null(client):
    rows = {table: None for table in _all_rows()}
    with patch.object(substrate_observability_routes, "_engine", return_value=_fake_engine(rows)):
        r = client.get("/api/substrate/observability/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["self_state"] is None
    assert body["attention_broadcast"] is None
    assert body["curiosity"] is None
    assert body["hub_presence"] is None


def test_summary_section_failure_isolated(client):
    rows = _all_rows()
    rows["substrate_endogenous_curiosity_candidates"] = RuntimeError("relation does not exist")
    with patch.object(substrate_observability_routes, "_engine", return_value=_fake_engine(rows)):
        r = client.get("/api/substrate/observability/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["curiosity"] is None
    assert body["self_state"] is not None
    assert body["attention_broadcast"] is not None


def test_summary_without_postgres_uses_live_presence(client, monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    hub_presence.record_turn(now=1000.0)
    with patch.object(hub_presence, "time") as fake_time:
        fake_time.time.return_value = 1030.0
        r = client.get("/api/substrate/observability/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["self_state"] is None
    assert body["hub_presence"]["connection_health"] == "active"
    assert body["hub_presence"]["last_turn_age_sec"] == 30.0


def test_summary_curiosity_signals_capped_and_ranked(client):
    rows = _all_rows()
    rows["substrate_endogenous_curiosity_candidates"] = {
        "candidates_json": [
            {"signal_type": "t", "signal_strength": i / 10.0, "evidence_summary": f"gap {i}"}
            for i in range(8)
        ],
        "generated_at": _NOW,
    }
    with patch.object(substrate_observability_routes, "_engine", return_value=_fake_engine(rows)):
        r = client.get("/api/substrate/observability/summary")

    body = r.json()
    assert body["curiosity"]["gap_count"] == 8
    assert len(body["curiosity"]["signals"]) == 5
    assert body["curiosity"]["signals"][0]["evidence_summary"] == "gap 7"
