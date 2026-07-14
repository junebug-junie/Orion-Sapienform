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

from scripts import substrate_execution_dispatch_routes  # noqa: E402


def _sample_dispatch_frame() -> dict:
    return {
        "schema_version": "execution.dispatch.frame.v1",
        "frame_id": "execution.dispatch.frame:policy.frame:pf1:execution_dispatch_policy.v1",
        "generated_at": "2026-05-24T12:00:00+00:00",
        "source_policy_frame_id": "policy.frame:pf1:substrate_policy.v1",
        "source_proposal_frame_id": "proposal.frame:pf1:proposal_policy.v1",
        "source_self_state_id": "self.state:pf1",
        "execution_dispatch_policy_id": "execution_dispatch_policy.v1",
        "dispatch_mode": "dry_run",
        "candidates": [],
        "blocked_candidates": [],
        "dispatched_candidates": [],
        "dispatch_attempted": False,
        "dispatch_count": 0,
        "blocked_count": 0,
        "warnings": [],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_execution_dispatch_routes.router)
    return TestClient(app)


def test_latest_returns_frame(client) -> None:
    sample = _sample_dispatch_frame()
    with patch.object(
        substrate_execution_dispatch_routes,
        "_load_latest_dispatch_frame",
        return_value=substrate_execution_dispatch_routes.ExecutionDispatchFrameV1.model_validate(
            sample
        ),
    ):
        resp = client.get("/api/substrate/execution-dispatch/latest")
    assert resp.status_code == 200
    assert resp.json()["dispatch_mode"] == "dry_run"
    assert resp.json()["dispatch_attempted"] is False


def test_latest_not_found(client) -> None:
    with patch.object(
        substrate_execution_dispatch_routes,
        "_load_latest_dispatch_frame",
        return_value=None,
    ):
        resp = client.get("/api/substrate/execution-dispatch/latest")
    assert resp.status_code == 404


def _candidate(status: str, dispatch_id: str) -> dict:
    return {
        "dispatch_id": dispatch_id,
        "source_decision_id": f"pd:{dispatch_id}",
        "source_proposal_id": f"proposal:{dispatch_id}",
        "dispatch_status": status,
        "dispatch_mode": "dispatch_read_only",
        "dispatch_kind": "inspect",
        "target_id": "capability:orchestration",
        "target_kind": "capability",
        "risk_score": 0.05,
        "confidence_score": 0.9,
        **(
            {"dispatched_at": "2026-07-14T00:00:00+00:00", "result_ref": f"result:{dispatch_id}"}
            if status == "dispatched"
            else {}
        ),
    }


def test_latest_includes_status_summary_counts(client) -> None:
    sample = _sample_dispatch_frame()
    sample["candidates"] = [
        _candidate("dry_run", "d1"),
        _candidate("dry_run", "d2"),
        _candidate("prepared_for_dispatch", "d3"),
    ]
    sample["dispatched_candidates"] = [_candidate("dispatched", "d4")]
    sample["dispatch_count"] = 1

    with patch.object(
        substrate_execution_dispatch_routes,
        "_load_latest_dispatch_frame",
        return_value=substrate_execution_dispatch_routes.ExecutionDispatchFrameV1.model_validate(
            sample
        ),
    ):
        resp = client.get("/api/substrate/execution-dispatch/latest")

    assert resp.status_code == 200
    summary = resp.json()["status_summary"]
    assert summary == {
        "dispatched_count": 1,
        "prepared_for_dispatch_count": 1,
        "dry_run_count": 2,
    }


def test_latest_status_summary_all_zero_on_empty_frame(client) -> None:
    sample = _sample_dispatch_frame()
    with patch.object(
        substrate_execution_dispatch_routes,
        "_load_latest_dispatch_frame",
        return_value=substrate_execution_dispatch_routes.ExecutionDispatchFrameV1.model_validate(
            sample
        ),
    ):
        resp = client.get("/api/substrate/execution-dispatch/latest")
    assert resp.json()["status_summary"] == {
        "dispatched_count": 0,
        "prepared_for_dispatch_count": 0,
        "dry_run_count": 0,
    }
