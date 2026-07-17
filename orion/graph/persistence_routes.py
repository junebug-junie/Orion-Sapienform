"""Workload-keyed graph persistence routing table."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Literal, Mapping

logger = logging.getLogger(__name__)

RouteTarget = Literal["falkor", "sparql", "rdf", "in_memory", "postgres", "disabled"]
ShadowTarget = RouteTarget | Literal["none"]

_ROUTE_TARGETS: frozenset[str] = frozenset(
    {"falkor", "sparql", "rdf", "in_memory", "postgres", "disabled"}
)
_SHADOW_TARGETS: frozenset[str] = _ROUTE_TARGETS | frozenset({"none"})

_ROUTE_ENV_KEY = "GRAPH_PERSISTENCE_ROUTES_JSON"


@dataclass(frozen=True)
class WorkloadRoute:
    primary: RouteTarget
    shadow: ShadowTarget = "none"


def _is_route_target(value: str) -> bool:
    return value in _ROUTE_TARGETS


def _is_shadow_target(value: str) -> bool:
    return value in _SHADOW_TARGETS


def _parse_route_entry(raw: object, workload: str) -> WorkloadRoute | None:
    if not isinstance(raw, dict):
        logger.warning(
            "graph_persistence_route_invalid workload=%s reason=not_object",
            workload,
        )
        return None

    primary_raw = raw.get("primary")
    if not isinstance(primary_raw, str) or not _is_route_target(primary_raw.strip().lower()):
        logger.warning(
            "graph_persistence_route_invalid workload=%s reason=invalid_primary",
            workload,
        )
        return None

    shadow_raw = raw.get("shadow", "none")
    if not isinstance(shadow_raw, str) or not _is_shadow_target(shadow_raw.strip().lower()):
        logger.warning(
            "graph_persistence_route_invalid workload=%s reason=invalid_shadow",
            workload,
        )
        return None

    return WorkloadRoute(
        primary=primary_raw.strip().lower(),  # type: ignore[arg-type]
        shadow=shadow_raw.strip().lower(),  # type: ignore[arg-type]
    )


def load_persistence_routes(
    environ: Mapping[str, str] | None = None,
) -> dict[str, WorkloadRoute]:
    """Parse ``GRAPH_PERSISTENCE_ROUTES_JSON`` into a workload → route map."""
    env = dict(os.environ if environ is None else environ)
    raw_json = (env.get(_ROUTE_ENV_KEY) or "").strip()
    if not raw_json:
        return {}

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("graph_persistence_routes_json_invalid reason=decode_error")
        return {}

    if not isinstance(parsed, dict):
        logger.warning("graph_persistence_routes_json_invalid reason=not_object")
        return {}

    routes: dict[str, WorkloadRoute] = {}
    for workload, entry in parsed.items():
        if not isinstance(workload, str) or not workload.strip():
            continue
        route = _parse_route_entry(entry, workload)
        if route is not None:
            routes[workload.strip()] = route
    return routes


def resolve_workload_route(
    workload: str,
    *,
    environ: Mapping[str, str] | None = None,
    routes: Mapping[str, WorkloadRoute] | None = None,
) -> WorkloadRoute:
    """Resolve a workload to its configured route, or ``disabled`` when unknown."""
    table = load_persistence_routes(environ) if routes is None else routes
    route = table.get(workload)
    if route is None:
        logger.warning(
            "graph_persistence_route_unknown workload=%s primary=disabled",
            workload,
        )
        return WorkloadRoute(primary="disabled", shadow="none")
    return route
