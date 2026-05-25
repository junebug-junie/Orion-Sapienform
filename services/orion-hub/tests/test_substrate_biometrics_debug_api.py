from __future__ import annotations

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

from scripts import substrate_biometrics_routes  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_biometrics_routes.router)
    return TestClient(app)


def test_latest_chain_shape(client):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        sql = str(stmt)
        m = MagicMock()
        if "grammar_events" in sql and "trace_id" in sql:
            m.mappings.return_value.first.return_value = {
                "trace_id": "biometrics.node:atlas:2026-05-24T12:00:00Z",
            }
        elif "substrate_organ_emissions" in sql:
            m.mappings.return_value.first.return_value = {
                "emission_json": {"emission_id": "oem_1", "organ_id": "biometrics_pressure"},
            }
        elif "substrate_reduction_receipts" in sql:
            m.mappings.return_value.first.return_value = {
                "receipt_json": {"receipt_id": "rcpt_1", "accepted_event_ids": ["gev_1"]},
            }
        elif "substrate_node_biometrics_projection" in sql:
            m.mappings.return_value.first.return_value = {
                "projection_json": {
                    "projection_id": "node_biometrics_projection",
                    "nodes": {
                        "atlas": {"node_id": "atlas", "availability_status": "online"},
                    },
                },
            }
        elif "substrate_active_node_pressure_projection" in sql:
            m.mappings.return_value.first.return_value = {
                "projection_json": {
                    "projection_id": "active_node_pressure_projection",
                    "nodes": {"atlas": {"node_id": "atlas", "pressure_score": 0.5}},
                },
            }
        else:
            m.mappings.return_value.first.return_value = None
        return m

    conn.execute.side_effect = execute

    with patch.object(substrate_biometrics_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/biometrics-node/atlas/latest")

    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == "atlas"
    assert "event_chain" in body
    assert len(body["event_chain"]) == 6
    assert body["latest_reduction_receipt"]["receipt_id"] == "rcpt_1"
