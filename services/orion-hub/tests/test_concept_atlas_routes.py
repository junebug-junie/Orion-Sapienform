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

from types import SimpleNamespace

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


def _temporal(observed_at=None):
    from orion.core.schemas.cognitive_substrate import SubstrateTemporalWindowV1

    return SubstrateTemporalWindowV1(observed_at=observed_at or datetime.now(timezone.utc))


def _concept_node(
    node_id,
    label,
    *,
    anchor_scope="orion",
    promotion_state="proposed",
    activation=0.0,
    decay_floor=0.0,
    observed_at=None,
    metadata=None,
):
    from orion.core.schemas.cognitive_substrate import ConceptNodeV1, SubstrateActivationV1, SubstrateSignalBundleV1

    return ConceptNodeV1(
        node_id=node_id,
        label=label,
        anchor_scope=anchor_scope,
        promotion_state=promotion_state,
        temporal=_temporal(observed_at),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(
            confidence=0.7,
            salience=0.5,
            activation=SubstrateActivationV1(activation=activation, decay_floor=decay_floor),
        ),
        metadata=metadata or {},
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
    # All three seeded nodes were just created (observed_at ~= now) -> too
    # young for _AT_RISK_MIN_AGE_SECONDS, regardless of their activation.
    assert body["at_risk"] == []
    assert body["at_risk_note"]


def test_summary_at_risk_reported_for_old_low_activation_nodes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import timedelta

    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    old_enough = datetime.now(timezone.utc) - timedelta(hours=2)
    healthy = _concept_node("concept-healthy", "Healthy", activation=0.9, decay_floor=0.1, observed_at=old_enough)
    decaying = _concept_node("concept-decaying", "Decaying", activation=0.05, decay_floor=0.02, observed_at=old_enough)
    store.upsert_node(identity_key="concept:healthy", node=healthy)
    store.upsert_node(identity_key="concept:decaying", node=decaying)
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/summary")
    assert r.status_code == 200
    body = r.json()
    at_risk_ids = {row["node_id"] for row in body["at_risk"]}
    assert "concept-decaying" in at_risk_ids
    assert "concept-healthy" not in at_risk_ids


def test_summary_at_risk_excludes_freshly_born_low_salience_node(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: ConceptNodeV1 now auto-seeds activation=salience at
    # construction time, so a brand-new, low-salience concept could
    # otherwise show up as "at risk of decaying" on its very first tick --
    # it hasn't decayed at all, it just started low. The age gate must
    # exclude it even though its activation already sits at/under the
    # decay_floor + margin threshold.
    from scripts import concept_atlas_routes
    from orion.core.schemas.cognitive_substrate import ConceptNodeV1, SubstrateSignalBundleV1
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fresh_low_salience = ConceptNodeV1(
        node_id="concept-fresh-low",
        label="Fresh Low Salience",
        anchor_scope="world",
        temporal=_temporal(),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(confidence=0.5, salience=0.01),
    )
    store.upsert_node(identity_key="concept:fresh-low", node=fresh_low_salience)
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["at_risk"] == []


# --- graph=aitown query param (2026-08-20) -----------------------------------
# AI Town's own concept graph (SUBSTRATE_SEMANTIC_STORE_AITOWN) was written
# by the scheduler but reachable by zero GET route before this -- these
# tests confirm ?graph=aitown actually switches which store is read, per the
# design spec's own "first cut" suggestion, rather than always resolving to
# Orion's store regardless of the param.


def test_summary_graph_param_defaults_to_orion_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    orion_store = _build_store()
    aitown_store = _build_store()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: orion_store)
    monkeypatch.setattr(concept_atlas_routes, "_get_aitown_substrate_store", lambda: aitown_store)

    r = client.get("/api/substrate/concepts/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["graph"] == "orion"


def test_summary_graph_param_aitown_switches_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    orion_store = _build_store()  # 3 seeded nodes
    aitown_store = InMemorySubstrateGraphStore()  # empty
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: orion_store)
    monkeypatch.setattr(concept_atlas_routes, "_get_aitown_substrate_store", lambda: aitown_store)

    r = client.get("/api/substrate/concepts/summary", params={"graph": "aitown"})
    assert r.status_code == 200
    body = r.json()
    assert body["graph"] == "aitown"
    assert body["total_concepts"] == 0  # aitown_store is empty, orion_store is not


def test_summary_graph_param_unrecognized_value_degrades_to_orion(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import concept_atlas_routes

    orion_store = _build_store()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: orion_store)

    r = client.get("/api/substrate/concepts/summary", params={"graph": "not-a-real-graph"})
    assert r.status_code == 200
    body = r.json()
    assert body["graph"] == "orion"
    assert body["total_concepts"] == 3


def test_network_graph_param_aitown_switches_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    orion_store = _build_store()
    aitown_store = InMemorySubstrateGraphStore()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: orion_store)
    monkeypatch.setattr(concept_atlas_routes, "_get_aitown_substrate_store", lambda: aitown_store)

    r = client.get("/api/substrate/concepts/network", params={"graph": "aitown"})
    assert r.status_code == 200
    body = r.json()
    assert body["graph"] == "aitown"
    assert body["nodes"] == []


def test_network_graph_param_defaults_to_orion_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes

    orion_store = _build_store()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: orion_store)

    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    assert body["graph"] == "orion"
    assert len(body["nodes"]) == 3


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
    assert body["component_count"] == 0


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


def test_network_canonical_node_is_always_god_node_regardless_of_degree(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression, confirmed live 2026-08-22: pure-degree god-node ranking
    # buried the golden-seeded Orion/Juniper/Claude/relationship anchors
    # (orion/substrate/seed_concepts.yaml) behind whatever topic-foundry
    # cluster racked up the most same-day co_occurs_with edges -- an
    # artifact of day-bucket co-occurrence rewarding vocabulary ubiquity, not
    # a real signal of what's load-bearing to Orion's identity. Independent
    # of the 2026-08-20 landmark-connection design (which fixed these nodes'
    # isolation, giving them real degree) -- this fixes ranking still being
    # pure-degree once connected. A canonical node must be a god node even
    # with zero edges; a non-canonical high-degree node must still fill any
    # remaining top-N slots.
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    golden = _concept_node("concept-golden", "Orion", promotion_state="canonical")
    store.upsert_node(identity_key="concept:golden", node=golden)

    # _GOD_NODE_TOP_N is 5: 1 canonical node here leaves 4 remaining slots
    # for non-canonical nodes, ranked strictly by degree -- so 6 non-canonical
    # contenders (hub + 5 spokes) at distinct, deterministic degrees (via
    # distinct edge salience) guarantees the weakest one is left out, proving
    # the cutoff still applies to everything that isn't canonical.
    hub = _concept_node("concept-noisy-hub", "topic_7", promotion_state="proposed")
    store.upsert_node(identity_key="concept:noisy-hub", node=hub)
    for i, salience in enumerate([0.9, 0.7, 0.5, 0.3, 0.1]):
        spoke = _concept_node(f"concept-spoke-{i}", f"Spoke {i}", promotion_state="proposed")
        store.upsert_node(identity_key=f"concept:spoke-{i}", node=spoke)
        store.upsert_edge(
            identity_key=f"edge:hub-spoke-{i}",
            edge=_edge(f"edge-hub-spoke-{i}", "concept-noisy-hub", f"concept-spoke-{i}", salience=salience),
        )

    isolate = _concept_node("concept-isolate-2", "Isolate", promotion_state="proposed")
    store.upsert_node(identity_key="concept:isolate-2", node=isolate)

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    nodes_by_id = {n["id"]: n for n in body["nodes"]}

    # canonical, zero-degree -> still a god node.
    assert nodes_by_id["concept-golden"]["degree"] == 0
    assert nodes_by_id["concept-golden"]["god_node"] is True
    # non-canonical, highest real degree in the whole graph -> fills a
    # remaining slot ahead of every spoke.
    assert nodes_by_id["concept-noisy-hub"]["god_node"] is True
    # non-canonical, highest-degree spoke -> fills one of the 3 remaining
    # slots left after the hub takes one of the 4.
    assert nodes_by_id["concept-spoke-0"]["god_node"] is True
    # non-canonical, weakest real degree -> 6 non-canonical contenders for
    # only 4 remaining slots, so the weakest loses out. The cutoff still
    # applies to real organic hubs; it just no longer applies to canonical
    # golden anchors.
    assert nodes_by_id["concept-spoke-4"]["god_node"] is False
    # non-canonical, zero degree -> never a god node.
    assert nodes_by_id["concept-isolate-2"]["god_node"] is False


def test_network_topic_foundry_synthetic_label_flagged_honestly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # topic_foundry.py::_derive_label falls back to a bare "topic_<id>"
    # placeholder when a run produced neither a real label nor keywords --
    # non-blank, but not a human label. The network route must surface that
    # so the UI can render it honestly instead of indistinguishable from a
    # real named concept.
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    synthetic = _concept_node(
        "concept-synthetic",
        "topic_7",
        anchor_scope="world",
        metadata={"source": "orion-topic-foundry", "topic_id": 7},
    )
    real = _concept_node(
        "concept-real",
        "autonomy",
        anchor_scope="orion",
        metadata={"concept_id": "c-1"},
    )
    store.upsert_node(identity_key="concept:synthetic", node=synthetic)
    store.upsert_node(identity_key="concept:real", node=real)
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    nodes_by_id = {n["id"]: n for n in r.json()["nodes"]}

    assert nodes_by_id["concept-synthetic"]["origin"] == "topic_foundry"
    assert nodes_by_id["concept-synthetic"]["synthetic_label"] is True
    assert nodes_by_id["concept-real"]["origin"] == "concept"
    assert nodes_by_id["concept-real"]["synthetic_label"] is False


def test_network_connected_components_grouped_and_counted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readability gap from the 2026-08-18 design spec: cose's layout gives
    disconnected components no visual grouping. component_id/component_count
    are computed fresh per request (union-find over the same filtered
    node/edge lists the rest of the response is built from), not sourced
    from any precomputed job -- reuses the hub-and-spoke-plus-isolate shape
    already seeded for the god-node test above."""
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    hub = _concept_node("concept-orion", "Orion")
    store.upsert_node(identity_key="concept:orion", node=hub)
    for i in range(2):
        spoke = _concept_node(f"concept-spoke-{i}", f"Spoke {i}")
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
    nodes_by_id = {n["id"]: n for n in body["nodes"]}

    hub_component = nodes_by_id["concept-orion"]["component_id"]
    assert nodes_by_id["concept-spoke-0"]["component_id"] == hub_component
    assert nodes_by_id["concept-spoke-1"]["component_id"] == hub_component
    assert nodes_by_id["concept-isolate"]["component_id"] != hub_component
    assert body["component_count"] == 2


def test_network_passes_through_topic_foundry_cluster_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """topic_id lives in ConceptNodeV1.metadata (orion/substrate/adapters/topic_foundry.py
    writes it there, no dedicated schema field exists) -- confirm the network
    route surfaces it for a tagged node and reports None for an untagged one,
    rather than silently discarding it the way it did before this patch."""
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    tagged = _concept_node("concept-tagged", "Tagged", metadata={"topic_id": "cluster-7"})
    untagged = _concept_node("concept-untagged", "Untagged")
    store.upsert_node(identity_key="concept:tagged", node=tagged)
    store.upsert_node(identity_key="concept:untagged", node=untagged)

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    nodes_by_id = {n["id"]: n for n in body["nodes"]}
    assert nodes_by_id["concept-tagged"]["topic_id"] == "cluster-7"
    assert nodes_by_id["concept-untagged"]["topic_id"] is None


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


# --- entity-node hydration, added 2026-08-20 -------------------------------
# docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md
# store.query_concept_region() only ever returns concept-kind nodes; an edge
# to an off-slice non-concept node (most commonly an EntityNodeV1 mention)
# used to be silently dropped by the route's own AND-based edge filter --
# these tests cover the hydration pass that fixes it (and, as a side effect,
# makes the landmark-connection edges from topic_foundry.py actually visible).


def _entity_node(node_id, label, *, anchor_scope="world"):
    from orion.core.schemas.cognitive_substrate import EntityNodeV1, SubstrateSignalBundleV1

    return EntityNodeV1(
        node_id=node_id,
        label=label,
        anchor_scope=anchor_scope,
        promotion_state="proposed",
        temporal=_temporal(),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.0),
    )


def _entity_edge(edge_id, source_concept_id, target_entity_id, *, predicate="associated_with"):
    from orion.core.schemas.cognitive_substrate import NodeRefV1, SubstrateEdgeV1

    return SubstrateEdgeV1(
        edge_id=edge_id,
        source=NodeRefV1(node_id=source_concept_id, node_kind="concept"),
        target=NodeRefV1(node_id=target_entity_id, node_kind="entity"),
        predicate=predicate,
        temporal=_temporal(),
        confidence=0.6,
        salience=0.0,
        provenance=_provenance(),
    )


def test_network_hydrates_off_slice_entity_node_reached_by_edge(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    concept = _concept_node("concept-orion", "Orion", anchor_scope="orion")
    entity = _entity_node("entity-mention-orion", "Orion")
    store.upsert_node(identity_key="concept:orion", node=concept)
    store.upsert_node(identity_key="entity:orion-mention", node=entity)
    store.upsert_edge(
        identity_key="edge:concept-entity",
        edge=_entity_edge("edge-concept-entity", "concept-orion", "entity-mention-orion"),
    )

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    nodes_by_id = {n["id"]: n for n in body["nodes"]}

    # Without hydration this node would never appear -- store.query_concept_region()
    # never returns entity-kind nodes in `nodes` to begin with.
    assert "entity-mention-orion" in nodes_by_id
    assert nodes_by_id["entity-mention-orion"]["node_kind"] == "entity"

    edge_pairs = {(e["source"], e["target"]) for e in body["edges"]}
    assert ("concept-orion", "entity-mention-orion") in edge_pairs

    # The whole point: the concept and its formerly-invisible entity mention
    # now share one connected component instead of the edge being dropped.
    assert nodes_by_id["concept-orion"]["component_id"] == nodes_by_id["entity-mention-orion"]["component_id"]


def test_network_hydration_never_readmits_a_scope_filtered_concept_node(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concept node the explicit ?scope= filter excluded must stay
    excluded -- hydration only ever adds non-concept nodes, never smuggles a
    deliberately-filtered concept node back in via its edge."""
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    orion_scope_concept = _concept_node("concept-orion", "Orion", anchor_scope="orion")
    world_scope_concept = _concept_node("concept-world-topic", "World Topic", anchor_scope="world")
    store.upsert_node(identity_key="concept:orion", node=orion_scope_concept)
    store.upsert_node(identity_key="concept:world-topic", node=world_scope_concept)
    store.upsert_edge(
        identity_key="edge:orion-world",
        edge=_edge("edge-orion-world", "concept-orion", "concept-world-topic"),
    )

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network", params={"scope": "world"})
    assert r.status_code == 200
    body = r.json()
    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {"concept-world-topic"}  # concept-orion stays excluded, not hydrated back in


def test_network_hydration_bounded_by_cap(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    concept = _concept_node("concept-hub", "Hub")
    store.upsert_node(identity_key="concept:hub", node=concept)
    over_cap = concept_atlas_routes._NETWORK_HYDRATION_MAX_EXTRA_NODES + 10
    for i in range(over_cap):
        entity = _entity_node(f"entity-{i}", f"Entity {i}")
        store.upsert_node(identity_key=f"entity:{i}", node=entity)
        store.upsert_edge(
            identity_key=f"edge:hub-entity-{i}",
            edge=_entity_edge(f"edge-hub-entity-{i}", "concept-hub", f"entity-{i}"),
        )

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)
    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200
    body = r.json()
    entity_nodes = [n for n in body["nodes"] if n["node_kind"] == "entity"]
    assert len(entity_nodes) == concept_atlas_routes._NETWORK_HYDRATION_MAX_EXTRA_NODES


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


def test_summary_reads_hydrated_falkor_store(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes
    from orion.substrate.falkor_store import (
        FalkorSubstrateStore,
        FalkorSubstrateStoreConfig,
        RecordingFalkorClient,
    )

    falkor_client = RecordingFalkorClient(
        hydrate_rows=[
            {
                "node_id": "concept-native-atlas",
                "node_kind": "concept",
                "identity_key": "concept:native-atlas",
                "label": "Native Atlas",
                "definition": None,
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "canonical",
                "risk_tier": "low",
                "confidence": 0.8,
                "salience": 0.7,
                "activation": 0.5,
                "recency_score": 0.4,
                "decay_floor": 0.0,
                "decay_half_life_seconds": None,
                "observed_at": "2026-07-16T00:00:00+00:00",
                "valid_from": None,
                "valid_to": None,
                "provenance_authority": "local_inferred",
                "provenance_source_kind": "test",
                "provenance_source_channel": "test:concept_atlas",
                "provenance_producer": "test_concept_atlas_routes",
                "provenance_model_name": None,
                "provenance_correlation_id": None,
                "provenance_trace_id": None,
                "provenance_tier_rank": None,
            }
        ]
    )
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=falkor_client,
        hydrate=True,
    )
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["total_concepts"] == 1
    assert body["by_promotion_state"]["canonical"] == 1
    assert body["by_anchor_scope"]["orion"] == 1


# ---------------------------------------------------------------------------
# Readable labels in the network payload (branch 3, 2026-08-28)
# ---------------------------------------------------------------------------


def test_display_labels_names_evidence_after_the_concept_it_supports() -> None:
    """EvidenceNodeV1 has no `label` field at all, so the payload used to fall
    back to the raw node_id and the atlas rendered rows of
    `sub-evidence-topicfoundry-<uuid>-<n>`."""
    from scripts.concept_atlas_routes import _display_labels

    concept = SimpleNamespace(node_id="c1", node_kind="concept", label="Code Review Process")
    evidence = SimpleNamespace(node_id="e1", node_kind="evidence")
    edge = SimpleNamespace(
        predicate="supports",
        source=SimpleNamespace(node_id="e1"),
        target=SimpleNamespace(node_id="c1"),
    )

    labels = _display_labels([concept, evidence], [edge])
    assert labels["c1"] == "Code Review Process"
    assert labels["e1"] == "Evidence for Code Review Process"


def test_display_labels_falls_back_to_node_id_never_empty() -> None:
    from scripts.concept_atlas_routes import _display_labels

    orphan = SimpleNamespace(node_id="e-orphan", node_kind="evidence")
    blank = SimpleNamespace(node_id="c-blank", node_kind="concept", label="")
    labels = _display_labels([orphan, blank], [])
    assert labels["e-orphan"] == "e-orphan"
    assert labels["c-blank"] == "c-blank"
    assert all(v for v in labels.values())


def test_display_labels_leaves_concept_to_concept_supports_alone() -> None:
    """`supports` also runs concept -> concept after relation classification;
    a real concept must never be renamed 'Evidence for ...'."""
    from scripts.concept_atlas_routes import _display_labels

    a = SimpleNamespace(node_id="c1", node_kind="concept", label="Alpha")
    b = SimpleNamespace(node_id="c2", node_kind="concept", label="Beta")
    edge = SimpleNamespace(
        predicate="supports",
        source=SimpleNamespace(node_id="c1"),
        target=SimpleNamespace(node_id="c2"),
    )
    labels = _display_labels([a, b], [edge])
    assert labels["c1"] == "Alpha"
    assert labels["c2"] == "Beta"


def test_scheduler_startup_tick_defaults_on() -> None:
    """The loop used to sleep a full 86400s interval BEFORE its first tick, so
    it needed 24 unbroken hours of Hub uptime to fire once -- and never had.

    Asserts the CODE default, not the live settings singleton: hub Settings
    reads .env plus process env, and turning this key off is a documented,
    supported operator action -- it must not turn the suite red on their box.
    """
    from app.settings import Settings

    field = Settings.model_fields["SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_RUN_AT_STARTUP"]
    assert field.default is True


# --- origin / synthetic_label must survive hydration -------------------------
#
# Both fields gated on metadata["source"] == "orion-topic-foundry". `source` is
# not in falkor_codec's closed metadata allowlist, so under the live
# SUBSTRATE_STORE_BACKEND=falkor it does not survive a rehydrate (forced at
# most every snapshot_force_refresh_ceiling_sec). Every node read back had
# source=None, so `origin` was permanently "concept" and `synthetic_label`
# permanently False -- the latter meaning a genuinely unlabeled "topic_<id>"
# cluster rendered as if it were a real concept name, the exact dishonest
# label that field exists to prevent. provenance.producer IS a native column
# and does survive: confirmed live 2026-08-29, 43 concepts carry
# producer='topic_foundry_adapter'.


def _hydrated_topic_foundry_node(node_id, label):
    """A node as it comes back from Falkor: producer intact, metadata empty."""
    from orion.core.schemas.cognitive_substrate import (
        ConceptNodeV1,
        SubstrateActivationV1,
        SubstrateProvenanceV1,
        SubstrateSignalBundleV1,
    )

    return ConceptNodeV1(
        node_id=node_id,
        label=label,
        anchor_scope="world",
        promotion_state="proposed",
        temporal=_temporal(),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="topic_foundry.run_topic",
            source_channel="test",
            producer="topic_foundry_adapter",
        ),
        signals=SubstrateSignalBundleV1(
            confidence=0.7, salience=0.5, activation=SubstrateActivationV1(activation=0.4)
        ),
        metadata={},  # `source` did not survive hydration
    )


def test_origin_survives_hydration_via_provenance_producer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key="c:1", node=_hydrated_topic_foundry_node("c-1", "Home lab"))
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    body = client.get("/api/substrate/concepts/network").json()
    node = next(n for n in body["nodes"] if n["id"] == "c-1")
    assert node["origin"] == "topic_foundry", "metadata['source'] does not survive Falkor hydration"


def test_synthetic_label_survives_hydration_via_provenance_producer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key="c:2", node=_hydrated_topic_foundry_node("c-2", "topic_17"))
    store.upsert_node(identity_key="c:3", node=_hydrated_topic_foundry_node("c-3", "Home lab"))
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    body = client.get("/api/substrate/concepts/network").json()
    by_id = {n["id"]: n for n in body["nodes"]}
    assert by_id["c-2"]["synthetic_label"] is True, "a bare topic_<id> must not read as a real name"
    assert by_id["c-3"]["synthetic_label"] is False, "a real label must not be flagged synthetic"


def test_a_non_topic_foundry_node_is_not_claimed_by_the_producer_check(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widening the check must not relabel every node as topic-foundry."""
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: _build_store())
    body = client.get("/api/substrate/concepts/network").json()
    assert {n["origin"] for n in body["nodes"]} == {"concept"}


# --- the network route must survive a node kind with no `label` field --------
#
# Live 500 on 2026-08-30: `AttributeError: 'EvidenceNodeV1' object has no
# attribute 'label'`. The whole atlas went down.
#
# `synthetic_label` reads `.label`, and EvidenceNodeV1 has none. That only ever
# worked because the gate's LEFT side (metadata["source"]) was False for every
# node after hydration, so `and` short-circuited before touching `.label`.
# Widening the left side to provenance.producer -- correct in itself, and the
# fix for a different bug -- made it true for evidence nodes too, since every
# evidence node is written by topic_foundry_adapter. The short-circuit was
# load-bearing and nothing in the code or the tests said so.
#
# Every previous evidence test called `_display_labels` directly, so no test
# ever put an evidence node through the route. These do.


def _evidence_node(node_id, *, producer="topic_foundry_adapter"):
    from orion.core.schemas.cognitive_substrate import (
        EvidenceNodeV1,
        SubstrateActivationV1,
        SubstrateProvenanceV1,
        SubstrateSignalBundleV1,
    )

    return EvidenceNodeV1(
        node_id=node_id,
        evidence_type="chat_turn",
        content_ref="ref-1",
        anchor_scope="world",
        temporal=_temporal(),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="topic_foundry.run_topic",
            source_channel="test",
            producer=producer,
        ),
        signals=SubstrateSignalBundleV1(
            confidence=0.7, salience=0.5, activation=SubstrateActivationV1(activation=0.4)
        ),
        metadata={},
    )


def test_network_route_does_not_500_on_a_node_without_a_label_field(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    store = InMemorySubstrateGraphStore()
    store.upsert_node(
        identity_key="c:1", node=_hydrated_topic_foundry_node("c-1", "Code Review Process")
    )
    store.upsert_node(identity_key="e:1", node=_evidence_node("e-1"))
    store.upsert_edge(
        identity_key="edge:e1-c1",
        edge=_edge("edge-e1-c1", "e-1", "c-1", predicate="supports"),
    )
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200, r.text

    by_id = {n["id"]: n for n in r.json()["nodes"]}
    evidence = by_id["e-1"]
    assert evidence["synthetic_label"] is False, "a node with no label is not a synthetic label"
    assert evidence["label"] == "Evidence for Code Review Process"
    assert evidence["origin"] == "topic_foundry"


def test_synthetic_label_still_flags_a_bare_topic_id_concept(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label-less guard must not disarm the check it protects."""
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key="c:2", node=_hydrated_topic_foundry_node("c-2", "topic_17"))
    store.upsert_node(identity_key="e:2", node=_evidence_node("e-2"))
    # The route only hydrates a non-concept node that an edge reaches, so the
    # evidence node needs its supports edge to appear in the payload at all.
    store.upsert_edge(
        identity_key="edge:e2-c2",
        edge=_edge("edge-e2-c2", "e-2", "c-2", predicate="supports"),
    )
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    by_id = {n["id"]: n for n in client.get("/api/substrate/concepts/network").json()["nodes"]}
    assert by_id["c-2"]["synthetic_label"] is True
    assert by_id["e-2"]["synthetic_label"] is False


def test_every_payload_field_is_readable_for_every_durable_node_kind(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structural guard rather than one more example: build one node of each
    durable kind and assert the route renders them all. The 500 was a field
    access valid for one kind and absent on another, and only an all-kinds
    sweep catches the next one of those."""
    from orion.core.schemas.cognitive_substrate import (
        EntityNodeV1,
        SubstrateActivationV1,
        SubstrateProvenanceV1,
        SubstrateSignalBundleV1,
    )
    from orion.graph.analytics import GraphAnalytics  # noqa: F401  (import guard only)
    from orion.substrate.falkor_codec import DURABLE_NODE_KINDS
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    prov = SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="topic_foundry.run_topic",
        source_channel="test",
        producer="topic_foundry_adapter",
    )
    sig = SubstrateSignalBundleV1(
        confidence=0.7, salience=0.5, activation=SubstrateActivationV1(activation=0.4)
    )
    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key="c:x", node=_hydrated_topic_foundry_node("c-x", "Alpha"))
    store.upsert_node(identity_key="e:x", node=_evidence_node("e-x"))
    store.upsert_node(
        identity_key="n:x",
        node=EntityNodeV1(
            node_id="n-x",
            label="athena",
            entity_type="host",
            anchor_scope="world",
            temporal=_temporal(),
            provenance=prov,
            signals=sig,
        ),
    )
    # every kind the store can persist must be represented above
    assert set(DURABLE_NODE_KINDS) == {"concept", "evidence", "entity"}
    store.upsert_edge(
        identity_key="edge:ex-cx", edge=_edge("edge-ex-cx", "e-x", "c-x", predicate="supports")
    )
    store.upsert_edge(
        identity_key="edge:cx-nx",
        edge=_edge("edge-cx-nx", "c-x", "n-x", predicate="associated_with"),
    )
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    r = client.get("/api/substrate/concepts/network")
    assert r.status_code == 200, r.text
    nodes = r.json()["nodes"]
    assert {n["node_kind"] for n in nodes} == {"concept", "evidence", "entity"}
    for node in nodes:
        assert node["label"], f"every node needs a readable label: {node}"
        assert node["origin"] in ("topic_foundry", "concept")
        assert isinstance(node["synthetic_label"], bool)


# --- hydration truncation must mean "we actually dropped something" ---------


def _evidence_with_edge(store, index, concept_id):
    ev = _evidence_node(f"e-hyd-{index}")
    store.upsert_node(identity_key=f"e:hyd:{index}", node=ev)
    store.upsert_edge(
        identity_key=f"edge:hyd:{index}",
        edge=_edge(f"edge-hyd-{index}", f"e-hyd-{index}", concept_id, predicate="supports"),
    )


def test_hydration_truncation_is_not_claimed_when_nothing_was_dropped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hydrated_count >= cap` is True when exactly `cap` candidates existed and
    ALL of them were hydrated -- nothing was cut, but the UI printed
    "view truncated". A warning that fires on an untruncated view erodes the
    trust this reporting exists to earn."""
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_NETWORK_HYDRATION_MAX_EXTRA_NODES", 2)
    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key="c:h", node=_hydrated_topic_foundry_node("c-h", "Anchor"))
    for i in range(2):  # exactly the cap
        _evidence_with_edge(store, i, "c-h")
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    body = client.get("/api/substrate/concepts/network").json()
    assert body["hydrated_count"] == 2
    assert body["hydration_truncated"] is False, "exactly-full is not truncated"


def test_hydration_truncation_is_reported_when_something_was_dropped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_NETWORK_HYDRATION_MAX_EXTRA_NODES", 2)
    store = InMemorySubstrateGraphStore()
    store.upsert_node(identity_key="c:h", node=_hydrated_topic_foundry_node("c-h", "Anchor"))
    for i in range(5):  # more than the cap
        _evidence_with_edge(store, i, "c-h")
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)

    body = client.get("/api/substrate/concepts/network").json()
    assert body["hydrated_count"] == 2
    assert body["hydration_truncated"] is True
