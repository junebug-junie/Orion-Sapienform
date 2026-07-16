"""Concept Atlas read routes (Phase 8 of the concept-graph-pipeline design).

Mirrors the isolated-router testing convention used by
``test_grammar_atlas_api.py``: build a minimal FastAPI app that only includes
``concept_atlas_routes.router`` and monkeypatch the module's store resolver
directly, rather than pulling in the full ``scripts.main`` app.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
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


def _concept_atlas_test_app() -> FastAPI:
    from scripts.concept_atlas_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client() -> TestClient:
    _ensure_hub_scripts_import_path()
    return TestClient(_concept_atlas_test_app())


def _provenance():
    from orion.core.schemas.cognitive_substrate import SubstrateProvenanceV1

    return SubstrateProvenanceV1(
        authority="human_verified",
        source_kind="test_fixture",
        source_channel="test:concept_atlas",
        producer="test_concept_atlas_routes",
    )


def _temporal():
    from orion.core.schemas.cognitive_substrate import SubstrateTemporalWindowV1

    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def _concept_node(node_id, label, *, anchor_scope="orion", promotion_state="proposed", activation=0.0, decay_floor=0.0):
    from orion.core.schemas.cognitive_substrate import ConceptNodeV1, SubstrateActivationV1, SubstrateSignalBundleV1

    return ConceptNodeV1(
        node_id=node_id,
        label=label,
        anchor_scope=anchor_scope,
        promotion_state=promotion_state,
        temporal=_temporal(),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(
            confidence=0.7,
            salience=0.5,
            activation=SubstrateActivationV1(activation=activation, decay_floor=decay_floor),
        ),
    )


def _edge(edge_id, source_id, target_id, *, predicate="co_occurs_with", salience=0.5):
    from orion.core.schemas.cognitive_substrate import NodeRefV1, SubstrateEdgeV1

    return SubstrateEdgeV1(
        edge_id=edge_id,
        source=NodeRefV1(node_id=source_id, node_kind="concept"),
        target=NodeRefV1(node_id=target_id, node_kind="concept"),
        predicate=predicate,
        temporal=_temporal(),
        confidence=0.6,
        salience=salience,
        provenance=_provenance(),
    )


def _build_store():
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    orion = _concept_node("concept-orion", "Orion", anchor_scope="orion", promotion_state="canonical")
    juniper = _concept_node("concept-juniper", "Juniper", anchor_scope="juniper", promotion_state="canonical")
    misc = _concept_node("concept-misc", "Misc topic", anchor_scope="world", promotion_state="proposed")
    store.upsert_node(identity_key="concept:orion", node=orion)
    store.upsert_node(identity_key="concept:juniper", node=juniper)
    store.upsert_node(identity_key="concept:misc", node=misc)
    store.upsert_edge(
        identity_key="edge:orion-juniper",
        edge=_edge("edge-orion-juniper", "concept-orion", "concept-juniper", predicate="co_occurs_with"),
    )
    store.upsert_edge(
        identity_key="edge:orion-misc",
        edge=_edge("edge-orion-misc", "concept-orion", "concept-misc", predicate="contradicts"),
    )
    return store


# --- summary ---------------------------------------------------------------


def test_summary_empty_store_degrades_gracefully(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: None)
    r = client.get("/api/substrate/concepts/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["total_concepts"] == 0
    assert body["by_promotion_state"] == {}
    assert body["at_risk"] == []


def test_summary_counts_with_seeded_nodes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    store = _build_store()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["total_concepts"] == 3
    assert body["by_promotion_state"]["canonical"] == 2
    assert body["by_promotion_state"]["proposed"] == 1
    assert body["by_anchor_scope"]["orion"] == 1
    assert body["by_anchor_scope"]["juniper"] == 1
    assert body["by_anchor_scope"]["world"] == 1
    assert body["edge_counts_by_predicate"]["co_occurs_with"] == 1
    assert body["edge_counts_by_predicate"]["contradicts"] == 1
    # All three seeded nodes share activation=0.0 -> no fabricated at_risk signal.
    assert body["at_risk"] == []
    assert body["at_risk_note"]


def test_summary_at_risk_reported_when_activation_has_real_variance(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    healthy = _concept_node("concept-healthy", "Healthy", activation=0.9, decay_floor=0.1)
    decaying = _concept_node("concept-decaying", "Decaying", activation=0.05, decay_floor=0.02)
    store.upsert_node(identity_key="concept:healthy", node=healthy)
    store.upsert_node(identity_key="concept:decaying", node=decaying)
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/summary")
    assert r.status_code == 200
    body = r.json()
    at_risk_ids = {row["node_id"] for row in body["at_risk"]}
    assert "concept-decaying" in at_risk_ids
    assert "concept-healthy" not in at_risk_ids


# --- network -----------------------------------------------------------------


def test_network_empty_store_degrades_gracefully(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: None)
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["nodes"] == []
    assert body["edges"] == []


def test_network_god_node_flag_on_highest_degree_node(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    # A small hub-and-spoke graph plus one isolated node: "orion" touches every
    # edge (highest degree, must be flagged), the isolated node touches none
    # (must never be flagged regardless of the top-N cutoff).
    store = InMemorySubstrateGraphStore()
    hub = _concept_node("concept-orion", "Orion")
    store.upsert_node(identity_key="concept:orion", node=hub)
    spokes = []
    for i in range(4):
        spoke = _concept_node(f"concept-spoke-{i}", f"Spoke {i}")
        spokes.append(spoke)
        store.upsert_node(identity_key=f"concept:spoke-{i}", node=spoke)
        store.upsert_edge(
            identity_key=f"edge:orion-spoke-{i}",
            edge=_edge(f"edge-orion-spoke-{i}", "concept-orion", f"concept-spoke-{i}"),
        )
    isolate = _concept_node("concept-isolate", "Isolate")
    store.upsert_node(identity_key="concept:isolate", node=isolate)

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    nodes_by_id = {n["id"]: n for n in body["nodes"]}
    assert nodes_by_id["concept-orion"]["god_node"] is True
    assert nodes_by_id["concept-orion"]["degree"] == pytest.approx(4 * 1.5)
    assert nodes_by_id["concept-isolate"]["god_node"] is False
    assert nodes_by_id["concept-isolate"]["degree"] == 0
    assert body["god_node_count"] >= 1


def test_network_malformed_query_params_do_not_500(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    store = _build_store()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get(
        "/api/substrate/concepts/network",
        params={"scope": "not-a-real-scope", "min_activation": "not-a-number", "focus": "does-not-exist"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    # Bad scope/min_activation are ignored (no-op), bad focus is ignored too --
    # all three seeded nodes should still be present, not filtered to nothing.
    assert len(body["nodes"]) == 3


def test_network_focus_filters_to_neighborhood(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    store = _build_store()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network", params={"focus": "Juniper"})
    assert r.status_code == 200
    body = r.json()
    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {"concept-orion", "concept-juniper"}


def test_network_nan_min_activation_is_ignored_not_silently_emptied(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """float('nan') parses without raising ValueError, but nan comparisons are
    always False -- without an explicit range check this would silently
    filter out every node instead of being treated as malformed input."""
    from scripts import concept_atlas_routes

    store = _build_store()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network", params={"min_activation": "nan"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert len(body["nodes"]) == 3


def test_network_surfaces_degraded_backend_result(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.store import SubstrateNeighborhoodSliceV1, SubstrateQueryResultV1

    class _DegradedStore:
        def query_concept_region(self, **kwargs):
            return SubstrateQueryResultV1(
                query_kind="concept_region",
                slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
                source_kind="graphdb",
                degraded=True,
                error="sparql_timeout",
            )

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: _DegradedStore())
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["degraded"] is True
    assert body["degraded_error"] == "sparql_timeout"


def test_network_store_error_degrades_gracefully(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    class _ExplodingStore:
        def query_concept_region(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: _ExplodingStore())
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["nodes"] == []


# --- page route + template/static-asset wiring --------------------------------


def test_concept_atlas_page_renders(client: TestClient) -> None:
    r = client.get("/concept-atlas")
    assert r.status_code == 200
    assert "concept-atlas.js" in r.text
    assert "OrionConceptAtlas" not in r.text  # that symbol lives in the JS file, not the template


def test_concept_atlas_template_references_correct_static_asset() -> None:
    template_path = HUB_ROOT / "templates" / "concept_atlas.html"
    js_path = HUB_ROOT / "static" / "js" / "concept-atlas.js"
    assert template_path.is_file()
    assert js_path.is_file()
    template_text = template_path.read_text(encoding="utf-8")
    assert "/static/js/concept-atlas.js" in template_text
    assert "cytoscape" in template_text.lower()


def test_concept_atlas_js_exposes_expected_namespace() -> None:
    js_path = HUB_ROOT / "static" / "js" / "concept-atlas.js"
    js_text = js_path.read_text(encoding="utf-8")
    assert "window.OrionConceptAtlas" in js_text
    assert "activate" in js_text
    assert "deactivate" in js_text


def test_index_html_wires_concept_atlas_tab() -> None:
    index_path = HUB_ROOT / "templates" / "index.html"
    index_text = index_path.read_text(encoding="utf-8")
    assert 'id="conceptAtlasTabButton"' in index_text
    assert 'data-panel="concept-atlas"' in index_text
    assert 'id="conceptAtlasPanelFrame"' in index_text
    assert 'src="/concept-atlas"' in index_text


def test_app_js_pings_activate_and_deactivate_for_concept_atlas() -> None:
    app_js_path = HUB_ROOT / "static" / "js" / "app.js"
    app_js_text = app_js_path.read_text(encoding="utf-8")
    assert "conceptAtlasPanelFrame" in app_js_text
    assert "OrionConceptAtlas.activate" in app_js_text
    assert "OrionConceptAtlas.deactivate" in app_js_text
