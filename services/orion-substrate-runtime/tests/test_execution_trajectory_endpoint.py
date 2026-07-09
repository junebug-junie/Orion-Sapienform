from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from orion.schemas.execution_projection import ExecutionRunStateV1, ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.reducer import reduce_execution_trace_events

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


@pytest.mark.asyncio
async def test_execution_trajectory_endpoint_serves_capped_runs(monkeypatch) -> None:
    main = _import_main(monkeypatch)

    base_ts = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    projection = ExecutionTrajectoryProjectionV1(
        projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
        generated_at=base_ts,
        runs={},
    )
    for i in range(5):
        ts = base_ts.replace(second=i)
        trace_id = f"cortex.exec:athena:corr-{i}"
        atom = GrammarAtomV1(
            atom_id=f"{trace_id}:exec_result_emitted",
            trace_id=trace_id,
            atom_type="observation",
            semantic_role="exec_result_emitted",
            layer="execution",
            summary="Cortex exec result emitted to reply_to=True, status=success",
        )
        event = GrammarEventV1(
            event_id=f"gev_{i}",
            event_kind="atom_emitted",
            trace_id=trace_id,
            emitted_at=ts,
            observed_at=ts,
            atom=atom,
            provenance=GrammarProvenanceV1(
                source_service="orion-cortex-exec",
                source_component="cortex_exec_grammar_emit",
            ),
            correlation_id=f"corr-{i}",
        )
        projection, _ = reduce_execution_trace_events(
            events=[event],
            projection=projection,
            now=ts,
            max_runs=3,
        )

    assert len(projection.runs) <= 3

    store = MagicMock()
    store.load_execution_trajectory.return_value = projection
    monkeypatch.setattr(main.worker, "_store", store, raising=False)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/projections/execution_trajectory")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert len(body["projection"]["runs"]) <= 3
