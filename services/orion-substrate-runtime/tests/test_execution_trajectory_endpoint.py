from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from orion.schemas.execution_projection import ExecutionRunStateV1, ExecutionTrajectoryProjectionV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID

REPO_ROOT = Path(__file__).resolve().parents[3]


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


@pytest.mark.asyncio
async def test_execution_trajectory_endpoint_returns_projection(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    now = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    proj = ExecutionTrajectoryProjectionV1(
        projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
        generated_at=now,
        runs={
            "trace-1": ExecutionRunStateV1(
                trace_id="trace-1",
                correlation_id="corr-1",
                node_id="athena",
                reasoning_present=True,
                recall_observed=True,
                step_count=4,
                failed_step_count=1,
                pressure_hints={"execution_friction": 0.2},
                last_updated_at=now,
            )
        },
    )
    store = MagicMock()
    store.load_execution_trajectory.return_value = proj
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/projections/execution_trajectory")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["projection"]["runs"]["trace-1"]["reasoning_present"] is True
    store.load_execution_trajectory.assert_called_once_with(EXECUTION_TRAJECTORY_PROJECTION_ID)


@pytest.mark.asyncio
async def test_execution_trajectory_endpoint_no_projection(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    store = MagicMock()
    store.load_execution_trajectory.return_value = None
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/projections/execution_trajectory")

    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "reason": "no_projection"}
