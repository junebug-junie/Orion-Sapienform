from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.route_projection import RouteArbitrationProjectionV1

from .constants import ROUTE_ARBITRATION_PROJECTION_ID


def empty_route_projection(*, now: datetime | None = None) -> RouteArbitrationProjectionV1:
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    return RouteArbitrationProjectionV1(
        projection_id=ROUTE_ARBITRATION_PROJECTION_ID,
        generated_at=clock,
        runs={},
    )
