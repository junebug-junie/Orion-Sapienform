"""GET /api/substrate/concepts/neighborhood and /path -- engine-side drill-down.

WHY THESE ARE NOT FILTERS ON /network. The network route fetches
`query_concept_region(limit_nodes=300, limit_edges=600)` and its `focus`
parameter filters that already-truncated slice, so it answers "neighbours among
the ones we happened to fetch". Measured live 2026-08-30 the EDGE cap binds:
600 of 1464 edges come back, and the canvas renders 102 of 671 nodes with zero
of the 464 entity nodes. These routes ask the engine over the whole graph.

Driven with a stubbed GraphAnalytics so the assertions are about what the ROUTE
derives, not about what a live graph happens to hold today.
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


def _stub():
    """Reproduces the live resolver's real behaviour, including ambiguity."""
    from orion.graph.analytics import RankedNode

    class StubAnalytics:
        def __init__(self):
            self.neighborhood_calls = []
            self.path_calls = []
            self.rel_types_seen = []

        def resolve_node(self, value, *, limit=5):
            v = str(value).strip()
            if v in ("Orion", "sub-concept-seed-orion"):
                return (RankedNode("sub-concept-seed-orion", "Orion", 0.0),)
            if v == "Juniper":
                return (RankedNode("sub-concept-seed-juniper", "Juniper", 1.0),)
            if v == "Hospital":  # three real concepts share this prefix live
                return (
                    RankedNode("c-h1", "Hospital and family chaos", 2.0),
                    RankedNode("c-h2", "Hospital and fear experience", 2.0),
                    RankedNode("c-h3", "Hospital and medical concerns", 2.0),
                )
            return ()

        def neighborhood(self, node_id, *, depth=1, rel_types=None, limit=200):
            self.neighborhood_calls.append((node_id, depth, rel_types, limit))
            self.rel_types_seen.append(rel_types)
            if rel_types and any(not str(t).replace("_", "").isalnum() for t in rel_types):
                raise ValueError(f"refusing to build Cypher for non-identifier relationship types: {rel_types!r}")
            return tuple(
                RankedNode(f"n-{i}", f"Neighbour {i}", float(1 + (i % depth)))
                for i in range(min(limit, 5))
            )

        def path(self, a, b, *, max_depth=3, rel_types=None):
            self.path_calls.append((a, b, max_depth, rel_types))
            if b == "sub-concept-seed-juniper":
                return (
                    RankedNode("sub-concept-seed-orion", "Orion", 0.0),
                    RankedNode("c-mid", "Sync Issue Resolution", 1.0),
                    RankedNode("sub-concept-seed-juniper", "Juniper", 2.0),
                )
            return ()

    return StubAnalytics()


@pytest.fixture
def stub(monkeypatch):
    import scripts.concept_atlas_routes as mod

    s = _stub()
    monkeypatch.setattr(mod, "_build_graph_analytics", lambda graph: (s, "substrate"))
    return s


# --- neighbourhood ----------------------------------------------------------


def test_neighborhood_resolves_a_label_and_returns_hops(client, stub):
    body = client.get("/api/substrate/concepts/neighborhood", params={"node": "Orion", "depth": 2}).json()
    assert body["available"] is True
    assert body["node_id"] == "sub-concept-seed-orion"
    assert body["depth"] == 2
    assert body["nodes"], "a resolved node must return its neighbourhood"
    assert all("hops" in n for n in body["nodes"])
    assert stub.neighborhood_calls[0][0] == "sub-concept-seed-orion", "the resolved id, not the raw text"


def test_neighborhood_refuses_to_guess_an_ambiguous_name(client, stub):
    """Labels are not unique -- three live concepts start with 'Hospital'.
    Silently taking the first would answer a question about a node the caller
    did not mean."""
    body = client.get("/api/substrate/concepts/neighborhood", params={"node": "Hospital"}).json()
    assert body["available"] is False
    assert body["reason"] == "node_ambiguous"
    assert len(body["candidates"]) == 3
    assert stub.neighborhood_calls == [], "must not traverse before the node is pinned down"


def test_neighborhood_distinguishes_not_found_from_ambiguous(client, stub):
    body = client.get("/api/substrate/concepts/neighborhood", params={"node": "nothing-like-this"}).json()
    assert body["available"] is False
    assert body["reason"] == "node_not_found"
    assert body["candidates"] == []


def test_neighborhood_requires_a_node(client, stub):
    body = client.get("/api/substrate/concepts/neighborhood").json()
    assert body["available"] is False
    assert body["reason"] == "node_required"


def test_neighborhood_reports_truncation(client, stub):
    """A neighbourhood cut short at the limit is a different fact from a small
    neighbourhood, and the list alone cannot tell them apart."""
    body = client.get(
        "/api/substrate/concepts/neighborhood", params={"node": "Orion", "limit": 5}
    ).json()
    assert body["truncated"] is True
    body2 = client.get(
        "/api/substrate/concepts/neighborhood", params={"node": "Orion", "limit": 50}
    ).json()
    assert body2["truncated"] is False


def test_neighborhood_passes_rel_types_through_and_rejects_injection(client, stub):
    client.get(
        "/api/substrate/concepts/neighborhood",
        params={"node": "Orion", "rel_types": "supports, associated_with"},
    )
    assert stub.rel_types_seen[-1] == ["supports", "associated_with"]

    body = client.get(
        "/api/substrate/concepts/neighborhood",
        params={"node": "Orion", "rel_types": "x) RETURN 1 //"},
    ).json()
    assert body["available"] is False
    assert body["reason"] == "invalid_rel_types"


def test_neighborhood_degrades_rather_than_500(client, monkeypatch):
    import scripts.concept_atlas_routes as mod

    class Exploding:
        def resolve_node(self, value, *, limit=5):
            raise RuntimeError("backend down")

    monkeypatch.setattr(mod, "_build_graph_analytics", lambda graph: (Exploding(), "substrate"))
    r = client.get("/api/substrate/concepts/neighborhood", params={"node": "Orion"})
    assert r.status_code == 200
    assert r.json()["reason"] == "graph_analytics_error"


# --- path -------------------------------------------------------------------


def test_path_returns_the_hop_chain(client, stub):
    body = client.get("/api/substrate/concepts/path", params={"from": "Orion", "to": "Juniper"}).json()
    assert body["available"] is True
    assert body["found"] is True
    assert [h["label"] for h in body["hops"]] == ["Orion", "Sync Issue Resolution", "Juniper"]
    assert [h["index"] for h in body["hops"]] == [0, 1, 2]


def test_an_empty_path_is_not_a_claim_of_disconnection(client, stub):
    """Empty means no path WITHIN max_depth. The payload has to say how far it
    looked, because the honest next step depends on which case it is."""
    body = client.get(
        "/api/substrate/concepts/path", params={"from": "Orion", "to": "sub-concept-seed-orion"}
    ).json()
    assert body["available"] is True
    assert body["found"] is False
    assert body["hops"] == []
    assert body["searched_to_depth"] >= 1


def test_path_refuses_an_ambiguous_endpoint(client, stub):
    body = client.get("/api/substrate/concepts/path", params={"from": "Orion", "to": "Hospital"}).json()
    assert body["available"] is False
    assert body["reason"] == "endpoint_not_resolved"
    assert len(body["to_candidates"]) == 3
    assert stub.path_calls == []


def test_path_requires_both_endpoints(client, stub):
    body = client.get("/api/substrate/concepts/path", params={"from": "Orion"}).json()
    assert body["available"] is False
    assert body["reason"] == "from_and_to_required"


def test_path_depth_is_capped(client, stub):
    """path() measured 384ms at depth 4 on the 671-node graph, up from 143ms at
    136 nodes -- the cap is a real cost control that has to hold as the graph
    grows."""
    from orion.graph.analytics import MAX_PATH_DEPTH

    body = client.get(
        "/api/substrate/concepts/path", params={"from": "Orion", "to": "Juniper", "max_depth": 99}
    ).json()
    assert body["max_depth"] == MAX_PATH_DEPTH
    assert stub.path_calls[-1][2] == 99, "the route passes it through; analytics clamps"


# --- both -------------------------------------------------------------------


def test_traversal_routes_are_unavailable_without_a_uri(client, monkeypatch):
    import scripts.concept_atlas_routes as mod

    monkeypatch.setattr(mod, "_build_graph_analytics", lambda graph: (None, "substrate"))
    for path in ("neighborhood", "path"):
        body = client.get(f"/api/substrate/concepts/{path}", params={"node": "x", "from": "a", "to": "b"}).json()
        assert body["available"] is False
        assert body["reason"] == "falkordb_uri_unset"


def test_the_blocking_traversal_runs_off_the_event_loop():
    """Same rule as /structure: these are synchronous Redis round trips, and
    path() is the most expensive call in the module."""
    _ensure_hub_scripts_import_path()
    import inspect

    import scripts.concept_atlas_routes as mod

    for fn in (mod.concept_atlas_neighborhood, mod.concept_atlas_path):
        assert "asyncio.to_thread" in inspect.getsource(fn), fn.__name__
