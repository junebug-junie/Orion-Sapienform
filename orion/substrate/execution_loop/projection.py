from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1

from .constants import EXECUTION_TRAJECTORY_PROJECTION_ID


def empty_execution_projection(*, now: datetime | None = None) -> ExecutionTrajectoryProjectionV1:
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    return ExecutionTrajectoryProjectionV1(
        projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
        generated_at=clock,
        runs={},
    )
