"""GET /api/substrate/concepts/structure -- the whole-graph structural read.

Mirrors test_concept_atlas_routes.py's isolated-router convention. The route is
driven with a stubbed GraphAnalytics rather than a live FalkorDB so the numbers
asserted here are the ones the route DERIVES, not ones a live graph happens to
hold today.

The live shape those stubs reproduce (orion_substrate, 2026-08-29): 136 nodes /
461 edges, 56 of them concepts, 12 components = 1 blob of 116 + 1 island of 10
+ 10 singletons, and 307 of the 461 edges are `co_occurs_with`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


@pytest.fixture
def client() -> TestClient:
    _ensure_hub_scripts_import_path()
    from scripts.concept_atlas_routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _live_shaped_analytics():
    """A stub GraphAnalytics reproducing the live orion_substrate shape."""
    from orion.graph.analytics import Component, RankedNode, StructureSummary

    class StubAnalytics:
        def __init__(self):
            self.rank_calls = []

        def summary(self):
            return StructureSummary(
                node_count=136,
                edge_count=461,
                edge_type_counts={"co_occurs_with": 307, "supports": 80, "associated_with": 74},
                components=(
                    Component("1", 116, ("Orion", "Juniper")),
                    Component("56", 10, ("Burst Test Conversations",)),
                    *[
                        Component(str(i), 1, (f"{name} prediction error",))
                        for i, name in enumerate(
                            ["Chat", "Execution", "Bus synaptic", "Codebase", "Perception",
                             "Route", "Biometrics", "Vision", "Transport", "Harness"],
                            start=4,
                        )
                    ],
                ),
            )

        def node_count(self, label=None):
            return 56 if label == "Concept" else 136

        def rank(self, measure, *, top_n=8):
            self.rank_calls.append((measure, top_n))
            if measure == "pagerank":
                return (
                    RankedNode("orion", "Orion", 0.0993),
                    RankedNode("juniper", "Juniper", 0.0964),
                    RankedNode("cr", "Code Review Process", 0.0339),
                    RankedNode("cd", "Code Debugging and Fixes", 0.0309),
                    RankedNode("amt", "AI Model Transparency", 0.0307),
                    RankedNode("mma", "messy middle authenticity", 0.0266),
                )
            if measure == "betweenness":
                return (
                    RankedNode("lfc", "Light folding concept", 85.75),
                    RankedNode("mma", "messy middle authenticity", 68.18),
                    RankedNode("hli", "Home lab infrastructure", 53.61),
                    RankedNode("orion", "Orion", 0.0),
                )
            return (RankedNode("cdw", "Chat Digest Workflow", 28.94),)

    return StubAnalytics()


@pytest.fixture
def stub_summary(monkeypatch):
    import scripts.concept_atlas_routes as mod

    stub = _live_shaped_analytics()
    monkeypatch.setattr(mod, "_build_graph_analytics", lambda graph: (stub, "substrate"))
    return stub


def test_summary_reports_the_whole_graph_shape(client, stub_summary):
    body = client.get("/api/substrate/concepts/structure").json()
    assert body["available"] is True
    assert body["graph"] == "substrate"
    assert body["node_count"] == 136
    assert body["concept_count"] == 56
    assert body["edge_count"] == 461
    assert body["component_count"] == 12
    assert body["largest_component_size"] == 116
    assert body["singleton_count"] == 10


def test_summary_names_the_dominant_edge_type_and_its_saturation(client, stub_summary):
    """307 co_occurs_with over 56 concepts = 307/1540 = 0.1994.

    The denominator is the CONCEPT count, not the node count: a concept-concept
    edge cannot land on an Evidence node, and using 136 would understate
    saturation roughly sixfold (461/9180 = 0.05) -- i.e. it would report a
    sparse graph where the real one is 20% saturated.
    """
    body = client.get("/api/substrate/concepts/structure").json()
    assert body["dominant_edge_type"] == "co_occurs_with"
    assert body["dominant_edge_saturation"] == pytest.approx(0.1994, abs=1e-4)
    graph_wide_would_be = 461 / (136 * 135 / 2)
    assert body["dominant_edge_saturation"] > graph_wide_would_be * 3


def test_bridges_surface_nodes_that_top_betweenness_but_not_pagerank(client, stub_summary):
    """The finding this route exists for: Orion and Juniper are pageRank #1/#2
    and are NOT the graph's bridges. `bridges` must exclude them and must
    surface the connectors that no other panel ranks."""
    body = client.get("/api/substrate/concepts/structure").json()
    labels = [b["label"] for b in body["bridges"]]
    assert "Light folding concept" in labels
    assert "messy middle authenticity" in labels
    assert "Orion" not in labels
    assert "Juniper" not in labels


def test_bridges_exclude_zero_scoring_nodes(client, stub_summary):
    """A node with betweenness 0.0 bridges nothing; listing it as a bridge
    would be a confidently wrong claim rather than a weak one."""
    body = client.get("/api/substrate/concepts/structure").json()
    assert all(b["score"] > 0.0 for b in body["bridges"])


def test_summary_lists_singleton_components_with_their_labels(client, stub_summary):
    body = client.get("/api/substrate/concepts/structure").json()
    singletons = [c for c in body["components"] if c["is_singleton"]]
    assert len(singletons) == 10
    assert any("prediction error" in (c["sample_labels"] or [""])[0] for c in singletons)


def test_summary_requests_all_three_measures(client, stub_summary):
    body = client.get("/api/substrate/concepts/structure").json()
    assert set(body["rankings"]) == {"pagerank", "betweenness", "harmonic"}
    assert {m for m, _ in stub_summary.rank_calls} == {"pagerank", "betweenness", "harmonic"}


def test_summary_is_unavailable_rather_than_500_when_the_uri_is_unset(client, monkeypatch):
    import scripts.concept_atlas_routes as mod

    monkeypatch.setattr(mod, "_build_graph_analytics", lambda graph: (None, "substrate"))
    r = client.get("/api/substrate/concepts/structure")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "falkordb_uri_unset"


def test_summary_degrades_rather_than_500_when_the_graph_errors(client, monkeypatch):
    """An operator page must never 500 on a backend hiccup -- the same rule the
    /network route already follows."""
    import scripts.concept_atlas_routes as mod
    from orion.graph.analytics import GraphAnalyticsError

    class Exploding:
        def summary(self):
            raise GraphAnalyticsError("graph query failed: backend down")

    monkeypatch.setattr(mod, "_build_graph_analytics", lambda graph: (Exploding(), "substrate"))
    r = client.get("/api/substrate/concepts/structure")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "graph_analytics_error"


def test_aitown_graph_param_resolves_to_the_same_default_as_the_store_builder():
    """orion/substrate/falkor_store.py::build_aitown_falkor_substrate_store_from_env
    defaults FALKORDB_AITOWN_SUBSTRATE_GRAPH to 'orion_substrate_aitown'. If this
    route disagreed it would silently summarise a different (or absent) graph
    than the one /network renders.
    """
    _ensure_hub_scripts_import_path()
    import inspect

    import scripts.concept_atlas_routes as mod
    from orion.substrate import falkor_store

    route_src = inspect.getsource(mod._build_graph_analytics)
    builder_src = inspect.getsource(falkor_store.build_aitown_falkor_substrate_store_from_env)
    assert 'graph_name_default="orion_substrate_aitown"' in builder_src
    assert '"orion_substrate_aitown"' in route_src


def test_no_two_routes_share_a_path_and_method():
    """A duplicate path does not raise in FastAPI -- it silently serves whichever
    route registered first, leaving the second dead with no error anywhere.

    This is not hypothetical: the structure route was first written as
    `/api/substrate/concepts/summary`, which this module already used for the
    stat-tile endpoint. The app started, the route existed in `router.routes`,
    and every request went to the older handler. A test asserting the payload
    caught it; nothing else would have.
    """
    _ensure_hub_scripts_import_path()
    from collections import Counter

    from scripts.concept_atlas_routes import router

    seen = Counter(
        (r.path, method)
        for r in router.routes
        for method in sorted(getattr(r, "methods", None) or [])
    )
    duplicates = {key: n for key, n in seen.items() if n > 1}
    assert duplicates == {}, f"duplicate route registrations shadow each other: {duplicates}"


def test_no_two_route_handlers_share_a_name():
    """The same shadowing, one layer up: two `async def concept_atlas_summary`
    in one module is legal Python -- the second rebinds the name while the
    router still holds a reference to the first function object.
    """
    _ensure_hub_scripts_import_path()
    from collections import Counter

    from scripts.concept_atlas_routes import router

    names = Counter(r.name for r in router.routes if getattr(r, "name", None))
    duplicates = {name: n for name, n in names.items() if n > 1}
    assert duplicates == {}, f"duplicate route handler names: {duplicates}"
