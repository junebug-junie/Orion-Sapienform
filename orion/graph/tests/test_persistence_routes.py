from __future__ import annotations

import json
import logging

import pytest

from orion.graph.persistence_router import GraphPersistenceRouter
from orion.graph.persistence_routes import (
    WorkloadRoute,
    load_persistence_routes,
    resolve_workload_route,
)


def test_load_persistence_routes_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    routes_json = {
        "substrate.drive_state": {"primary": "falkor", "shadow": "sparql"},
        "substrate.concept": {"primary": "falkor", "shadow": "none"},
    }
    monkeypatch.setenv("GRAPH_PERSISTENCE_ROUTES_JSON", json.dumps(routes_json))

    routes = load_persistence_routes()

    assert routes["substrate.drive_state"] == WorkloadRoute(
        primary="falkor", shadow="sparql"
    )
    assert routes["substrate.concept"] == WorkloadRoute(
        primary="falkor", shadow="none"
    )


def test_resolve_workload_route_known_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes_json = {
        "substrate.drive_state": {"primary": "falkor", "shadow": "sparql"},
    }
    monkeypatch.setenv("GRAPH_PERSISTENCE_ROUTES_JSON", json.dumps(routes_json))

    route = resolve_workload_route("substrate.drive_state")

    assert route.primary == "falkor"
    assert route.shadow == "sparql"


def test_resolve_workload_route_unknown_returns_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("GRAPH_PERSISTENCE_ROUTES_JSON", "{}")
    caplog.set_level(logging.WARNING)

    route = resolve_workload_route("substrate.missing")

    assert route == WorkloadRoute(primary="disabled", shadow="none")
    assert any(
        "graph_persistence_route_unknown" in record.message for record in caplog.records
    )


def test_resolve_workload_route_empty_env_returns_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPH_PERSISTENCE_ROUTES_JSON", raising=False)

    route = resolve_workload_route("substrate.drive_state")

    assert route == WorkloadRoute(primary="disabled", shadow="none")


def test_router_select_logs_route(
    caplog: pytest.LogCaptureFixture,
) -> None:
    routes = {
        "substrate.drive_state": WorkloadRoute(primary="falkor", shadow="sparql"),
    }
    router = GraphPersistenceRouter(routes=routes)
    caplog.set_level(logging.INFO)

    route = router.select("substrate.drive_state")

    assert route.primary == "falkor"
    assert route.shadow == "sparql"
    assert any(
        record.message
        == "graph_route_selected workload=substrate.drive_state primary=falkor shadow=sparql"
        for record in caplog.records
    )


def test_router_select_routing_hint_overrides_primary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    routes = {
        "substrate.drive_state": WorkloadRoute(primary="falkor", shadow="sparql"),
    }
    router = GraphPersistenceRouter(routes=routes)
    caplog.set_level(logging.INFO)

    route = router.select("substrate.drive_state", routing_hint="in_memory")

    assert route.primary == "in_memory"
    assert route.shadow == "sparql"


def test_router_select_unknown_workload_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = GraphPersistenceRouter(routes={})
    caplog.set_level(logging.WARNING)

    route = router.select("unknown.workload")

    assert route == WorkloadRoute(primary="disabled", shadow="none")
    assert any(
        "graph_persistence_route_unknown" in record.message for record in caplog.records
    )
