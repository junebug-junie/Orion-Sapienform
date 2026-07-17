"""Graph persistence router: workload → primary/shadow backend selection."""

from __future__ import annotations

import logging
from typing import Mapping

from orion.graph.persistence_routes import (
    RouteTarget,
    WorkloadRoute,
    _ROUTE_TARGETS,
    load_persistence_routes,
)

logger = logging.getLogger(__name__)


class GraphPersistenceRouter:
    """Select primary (+ optional shadow) persistence targets per workload."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        routes: Mapping[str, WorkloadRoute] | None = None,
    ) -> None:
        self._routes = (
            load_persistence_routes(environ) if routes is None else dict(routes)
        )

    def select(
        self,
        workload: str,
        routing_hint: str | None = None,
    ) -> WorkloadRoute:
        """Return the route for ``workload``, optionally overriding primary via hint."""
        route = self._routes.get(workload)
        if route is None:
            logger.warning(
                "graph_persistence_route_unknown workload=%s primary=disabled",
                workload,
            )
            route = WorkloadRoute(primary="disabled", shadow="none")

        primary: RouteTarget = route.primary
        if routing_hint:
            hint = routing_hint.strip().lower()
            if hint in _ROUTE_TARGETS:
                primary = hint  # type: ignore[assignment]
            else:
                logger.warning(
                    "graph_route_hint_invalid workload=%s hint=%s",
                    workload,
                    routing_hint,
                )

        selected = WorkloadRoute(primary=primary, shadow=route.shadow)
        logger.info(
            "graph_route_selected workload=%s primary=%s shadow=%s",
            workload,
            selected.primary,
            selected.shadow,
        )
        return selected
