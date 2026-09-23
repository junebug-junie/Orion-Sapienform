from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from orion.schemas.system_one_appraisal import (
    SystemOneAnswerV1,
    SystemOneAppraisalFrameV1,
    SystemOneInputStateV1,
    SystemOneQuestionV1,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)


def _import_main(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused:5432/unused")
    monkeypatch.setenv(
        "NODE_CATALOG_PATH",
        str(REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"),
    )
    import app.settings as settings_mod

    settings_mod._settings = None
    import app.main as main

    return main


def _frame() -> SystemOneAppraisalFrameV1:
    question = SystemOneQuestionV1(
        type="score",
        instructions="Rate",
        criteria=["low", "medium", "high"],
    )
    state = SystemOneInputStateV1(
        source_broadcast_projection_id="substrate.attention.broadcast.v1",
        source_broadcast_generated_at=NOW,
        selected_action_type="reflect",
        coalition_stability_score=0.7,
    )
    answer = SystemOneAnswerV1(
        question_id="reverie_fit",
        type="score",
        score=1.3,
        confidence=0.6,
        probabilities={"0": 0.1, "1": 0.5, "2": 0.4},
    )
    return SystemOneAppraisalFrameV1(
        frame_id="frame-1",
        question_set_id="test.v1",
        generated_at=NOW,
        expires_at=NOW + timedelta(seconds=90),
        provider="kev",
        model_id="kev-latest",
        input_state=state,
        questions={"reverie_fit": question},
        answers={"reverie_fit": answer},
    )


@pytest.mark.asyncio
async def test_system_one_appraisal_endpoint_returns_latest_frame(monkeypatch) -> None:
    main = _import_main(monkeypatch)
    store = MagicMock()
    store.load_latest_system_one_appraisal.return_value = _frame()
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/projections/system_one_appraisal")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["projection"]["schema_version"] == "system_one.appraisal.frame.v1"
    assert body["projection"]["answers"]["reverie_fit"]["score"] == 1.3
    store.load_latest_system_one_appraisal.assert_called_once_with()


@pytest.mark.asyncio
async def test_system_one_appraisal_endpoint_reports_absence(monkeypatch) -> None:
    main = _import_main(monkeypatch)
    store = MagicMock()
    store.load_latest_system_one_appraisal.return_value = None
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/projections/system_one_appraisal")

    assert response.status_code == 200
    assert response.json() == {"ok": False, "reason": "no_projection"}
