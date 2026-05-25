from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

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

from scripts import substrate_feedback_routes  # noqa: E402


def _sample_feedback_frame() -> dict:
    return {
        "schema_version": "feedback.frame.v1",
        "frame_id": "feedback.frame:execution.dispatch.frame:pf1:feedback_policy.v1",
        "generated_at": "2026-05-25T12:00:00+00:00",
        "source_execution_dispatch_frame_id": "execution.dispatch.frame:pf1:execution_dispatch_policy.v1",
        "source_policy_frame_id": "policy.frame:pf1:substrate_policy.v1",
        "feedback_policy_id": "feedback_policy.v1",
        "outcome_status": "dry_run_only",
        "outcome_score": 0.5,
        "confidence_score": 0.9,
        "observations": [],
        "positive_evidence": [],
        "negative_evidence": [],
        "absence_evidence": [],
        "pressure_before": {},
        "pressure_after": {},
        "pressure_delta": {},
        "warnings": [],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_feedback_routes.router)
    return TestClient(app)


def test_latest_returns_frame(client) -> None:
    sample = _sample_feedback_frame()
    with patch.object(
        substrate_feedback_routes,
        "_load_latest_feedback_frame",
        return_value=substrate_feedback_routes.FeedbackFrameV1.model_validate(sample),
    ):
        resp = client.get("/api/substrate/feedback/latest")
    assert resp.status_code == 200
    assert resp.json()["outcome_status"] == "dry_run_only"


def test_latest_not_found(client) -> None:
    with patch.object(
        substrate_feedback_routes,
        "_load_latest_feedback_frame",
        return_value=None,
    ):
        resp = client.get("/api/substrate/feedback/latest")
    assert resp.status_code == 404
