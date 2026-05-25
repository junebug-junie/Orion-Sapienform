from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.execution_projection import (
    ExecutionRunStateV1,
    ExecutionTrajectoryProjectionV1,
)


def test_execution_projection_roundtrip() -> None:
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    run = ExecutionRunStateV1(
        trace_id="cortex.exec:athena:corr-1",
        correlation_id="corr-1",
        node_id="athena",
        status="success",
        step_count=2,
        started_step_count=2,
        completed_step_count=2,
        failed_step_count=0,
        pressure_hints={"execution_load": 0.25},
        evidence_event_ids=["gev_1"],
        last_updated_at=now,
    )
    proj = ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory",
        generated_at=now,
        runs={"cortex.exec:athena:corr-1": run},
    )
    data = proj.model_dump(mode="json")
    assert (
        ExecutionTrajectoryProjectionV1.model_validate(data)
        .runs["cortex.exec:athena:corr-1"]
        .node_id
        == "athena"
    )
