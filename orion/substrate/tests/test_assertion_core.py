"""Shared assertion core (memory Stage 2 PR A): contracts, codec, reconcile fence,
cognitive isolation and the neighborhood's assertion-state gate. DB-free.

Real-store lanes: test_assertion_core_falkor.py (FalkorDB) and
test_graph_journal_pg.py (Postgres journal + projector).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from orion.core.schemas.cognitive_substrate import (
    AssertionNodeV1,
    ConceptNodeV1,
    EntityNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateActivationV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.attention_broadcast import substrate_pressure_signals
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.eligibility import cognitive_view, is_cognitive_node
from orion.substrate.falkor_codec import (
    decode_edge,
    decode_node,
    encode_edge_properties,
    encode_node_properties,
)
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.neighborhood import NeighborhoodRequestV1, walkable_edge
from orion.substrate.reconcile import SubstrateIdentityResolver, merge_edge, merge_node
from orion.substrate.store import InMemorySubstrateGraphStore

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
FENCED = "memory.referents"


def prov(producer: str = "topic_foundry_adapter", **kw) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(authority="local_inferred", source_kind="test", source_channel="test",
                                 producer=producer, **kw)


def entity(node_id: str, label: str, *, producer: str = "topic_foundry_adapter", state: str = "provisional",
           scope: str = "world", **kw) -> EntityNodeV1:
    return EntityNodeV1(node_id=node_id, label=label, anchor_scope=scope, promotion_state=state,
                        temporal=SubstrateTemporalWindowV1(observed_at=NOW), provenance=prov(producer), **kw)


def concept(node_id: str, label: str, *, producer: str = "topic_foundry_adapter", salience: float = 0.5,
            metadata: dict | None = None, state: str = "provisional") -> ConceptNodeV1:
    return ConceptNodeV1(node_id=node_id, label=label, anchor_scope="orion", promotion_state=state,
                         temporal=SubstrateTemporalWindowV1(observed_at=NOW - timedelta(seconds=60)),
                         signals=SubstrateSignalBundleV1(salience=salience, confidence=0.7,
                                                         activation=SubstrateActivationV1(activation=salience,
                                                                                          decay_half_life_seconds=3600)),
                         provenance=prov(producer), metadata=metadata or {})


def assertion(node_id: str = "assertion-1", *, state: str = "provisional", revision: int = 1) -> AssertionNodeV1:
    return AssertionNodeV1(node_id=node_id, anchor_scope="juniper", promotion_state=state,
                           temporal=SubstrateTemporalWindowV1(observed_at=NOW), provenance=prov(FENCED),
                           predicate="co_occurs_with", statement_key="ref-a|co_occurs_with|ref-b|",
                           statement_text="Quill and the spring retreat were named together", revision=revision,
                           decision_ref="dec-1")


def edge(edge_id: str, src: tuple[str, str], dst: tuple[str, str], predicate: str = "co_occurs_with", **kw):
    return SubstrateEdgeV1(edge_id=edge_id, source=NodeRefV1(node_id=src[0], node_kind=src[1]),
                           target=NodeRefV1(node_id=dst[0], node_kind=dst[1]), predicate=predicate,
                           temporal=kw.pop("temporal", SubstrateTemporalWindowV1(observed_at=NOW)),
                           confidence=kw.pop("confidence", 0.8), provenance=kw.pop("provenance", prov()), **kw)


# ── contract + codec ────────────────────────────────────────────────────────


def test_assertion_node_round_trips_through_the_codec():
    node = assertion(revision=3)
    props = encode_node_properties(node, "fenced|assertion|assertion-1")
    assert props["assertion_predicate"] == "co_occurs_with" and props["assertion_revision"] == 3
    decoded = decode_node(props)
    assert isinstance(decoded, AssertionNodeV1)
    assert decoded.model_dump() == node.model_dump()


def test_edge_role_and_assertion_fields_round_trip_and_old_rows_decode_as_legacy():
    projection = edge("e-p", ("n-a", "entity"), ("n-b", "entity"), edge_role="semantic_projection",
                      assertion_id="assertion-1", assertion_revision=2)
    row = encode_edge_properties(projection, "projection|assertion-1")
    assert decode_edge(row).model_dump() == projection.model_dump()
    legacy_row = {k: v for k, v in row.items() if k not in {"edge_role", "assertion_id", "assertion_revision"}}
    old = decode_edge(legacy_row)
    assert old.edge_role == "legacy_unreviewed" and old.assertion_id is None


def test_edge_role_validators_refuse_unverifiable_shapes():
    with pytest.raises(ValidationError, match="assertion_id and assertion_revision"):
        edge("e1", ("n-a", "entity"), ("n-b", "entity"), edge_role="semantic_projection")
    with pytest.raises(ValidationError, match="Concept/Entity endpoints"):
        edge("e2", ("n-a", "evidence"), ("n-b", "entity"), edge_role="semantic_projection",
             assertion_id="x", assertion_revision=1)
    with pytest.raises(ValidationError, match="only semantic_projection"):
        edge("e3", ("n-a", "entity"), ("n-b", "entity"), assertion_id="x", assertion_revision=1)
    with pytest.raises(ValidationError, match="from an Assertion"):
        edge("e4", ("n-a", "entity"), ("n-b", "entity"), predicate="assertion_subject", edge_role="assertion_structure")


# ── reconcile fence ─────────────────────────────────────────────────────────


def test_a_fenced_referent_never_merges_with_a_same_label_topic_entity():
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    topic = entity("topic-circe", "circe")
    materializer.apply_record(SubstrateGraphRecordV1(anchor_scope="world", nodes=[topic]))
    # Without the fence this incoming entity gets the same label key and merges.
    unfenced = entity("other-circe", "circe")
    assert materializer.apply_record(
        SubstrateGraphRecordV1(anchor_scope="world", nodes=[unfenced])).node_decisions[0].merged is True

    ours = entity("referent-circe", "circe", producer=FENCED)
    result = materializer.apply_record(SubstrateGraphRecordV1(anchor_scope="world", nodes=[ours]))
    assert result.node_decisions[0].merged is False
    assert store.get_node_by_id("referent-circe").label == "circe"
    assert store.get_node_by_id("topic-circe") is not None


def test_embedding_identity_never_crosses_the_fence_in_either_direction():
    store = InMemorySubstrateGraphStore()
    resolver = SubstrateIdentityResolver(store=store)
    vec = [0.1, 0.9, 0.3]
    fenced = concept("referent-space", "space", producer=FENCED, metadata={"concept_embedding": vec})
    store.upsert_node(identity_key=resolver.canonical_node_key(fenced), node=fenced)
    # An unfenced near-duplicate concept must not resolve onto the fenced node...
    incoming = concept("topic-space-2", "outer space", metadata={"concept_embedding": vec})
    assert resolver._concept_embedding_match_key(incoming, scope="orion", subject="") is None
    # ...while the same pair with no fence does merge (proves the test can fail).
    plain = concept("topic-space", "space", metadata={"concept_embedding": vec})
    store.upsert_node(identity_key=resolver.canonical_node_key(plain), node=plain)
    assert resolver._concept_embedding_match_key(incoming, scope="orion", subject="") == "concept|orion||label:space"
    # And a fenced concept never takes the embedding path at all.
    assert resolver.canonical_node_key(fenced) == "fenced|concept|referent-space"


def test_a_fenced_node_takes_its_projector_s_lifecycle_on_merge():
    old = assertion(state="provisional", revision=1)
    new = assertion(state="rejected", revision=2)
    merged = merge_node(old, new, source_graph_id="g")
    assert (merged.promotion_state, merged.revision) == ("rejected", 2)
    # Unfenced nodes keep the existing merge policy (state is not taken from incoming).
    a, b = entity("x", "x", state="provisional"), entity("x", "x", state="rejected")
    assert merge_node(a, b, source_graph_id="g").promotion_state == "provisional"


def test_a_role_bearing_edge_takes_the_projector_s_validity_but_keeps_its_id():
    opened = edge("e-prov", ("n-r", "entity"), ("n-ev", "evidence"), predicate="observed_in", edge_role="provenance")
    closed = edge("e-other", ("n-r", "entity"), ("n-ev", "evidence"), predicate="observed_in", edge_role="provenance",
                  temporal=SubstrateTemporalWindowV1(observed_at=NOW, valid_to=NOW + timedelta(days=1)))
    merged = merge_edge(opened, closed, source_graph_id="g")
    assert merged.edge_id == "e-prov" and merged.temporal.valid_to == NOW + timedelta(days=1)
    legacy = merge_edge(edge("l1", ("n-a", "concept"), ("n-b", "concept")),
                        edge("l2", ("n-a", "concept"), ("n-b", "concept"),
                             temporal=SubstrateTemporalWindowV1(observed_at=NOW, valid_to=NOW + timedelta(days=1))),
                        source_graph_id="g")
    assert legacy.temporal.valid_to is None  # legacy merge policy unchanged


def test_a_projection_never_merges_into_a_same_endpoint_legacy_edge():
    key = SubstrateIdentityResolver.canonical_edge_key
    legacy = edge("l", ("n-a", "concept"), ("n-b", "concept"))
    projection = edge("p", ("n-a", "concept"), ("n-b", "concept"), edge_role="semantic_projection",
                      assertion_id="assertion-1", assertion_revision=1)
    assert key(legacy) == "n-a|co_occurs_with|n-b"
    assert key(projection) == "projection|assertion-1"


# ── cognitive isolation (#2497 rule 8) ──────────────────────────────────────


def _legacy_graph(store: InMemorySubstrateGraphStore) -> None:
    store.upsert_node(identity_key="c:a", node=concept("n-a", "gpu", salience=0.9,
                                                       metadata={"prediction_error": 0.8}))
    store.upsert_node(identity_key="c:b", node=concept("n-b", "fall", salience=0.4))
    store.upsert_node(identity_key="c:c", node=concept("n-c", "camera", salience=0.1))
    store.upsert_edge(identity_key="n-a|associated_with|n-b", edge=edge("e1", ("n-a", "concept"), ("n-b", "concept"),
                                                                     predicate="associated_with"))
    store.upsert_edge(identity_key="b|supports|c", edge=edge("e2", ("n-b", "concept"), ("n-c", "concept"),
                                                              predicate="supports"))
    # dangling legacy edge: its treatment must not change either
    store.upsert_edge(identity_key="c|refines|gone", edge=edge("e3", ("n-c", "concept"), ("gone", "concept"),
                                                                predicate="refines"))


def _add_memory_structure(store: InMemorySubstrateGraphStore) -> None:
    store.upsert_node(identity_key="f:r1", node=entity("ref-r1", "hecate", producer=FENCED, scope="orion"))
    store.upsert_node(identity_key="f:ev", node=EvidenceNodeV1(
        node_id="ev1", evidence_type="episode_memory", content_ref="episode_memory:1", anchor_scope="juniper",
        temporal=SubstrateTemporalWindowV1(observed_at=NOW), provenance=prov(FENCED)))
    store.upsert_node(identity_key="f:as", node=assertion("as1"))
    store.upsert_edge(identity_key="p1", edge=edge("p1", ("ref-r1", "entity"), ("ev1", "evidence"),
                                                   predicate="observed_in", edge_role="provenance"))
    store.upsert_edge(identity_key="s1", edge=edge("s1", ("as1", "assertion"), ("n-a", "concept"),
                                                   predicate="assertion_subject", edge_role="assertion_structure"))
    store.upsert_edge(identity_key="proj", edge=edge("proj", ("n-a", "concept"), ("n-c", "concept"),
                                                     predicate="causes", edge_role="semantic_projection",
                                                     assertion_id="as1", assertion_revision=1))


def _dynamics(store):
    result = SubstrateDynamicsEngine(store=store).tick(now=NOW)
    legacy = {"n-a", "n-b", "n-c", "gone"}
    return (
        sorted((u.node_id, round(u.new_activation, 9), u.reason) for u in result.activation_updates
               if u.node_id in legacy),
        sorted((u.node_id, round(u.new_pressure, 9), u.reason) for u in result.pressure_updates
               if u.node_id in legacy),
        {u.node_id for u in result.activation_updates} | {u.node_id for u in result.pressure_updates},
    )


def test_dynamics_is_identical_before_and_after_projecting_memory_structure():
    before, after = InMemorySubstrateGraphStore(), InMemorySubstrateGraphStore()
    _legacy_graph(before)
    _legacy_graph(after)
    _add_memory_structure(after)
    b_act, b_pressure, _ = _dynamics(before)
    a_act, a_pressure, touched = _dynamics(after)
    assert b_pressure and b_act  # the fixture really moves
    assert (a_act, a_pressure) == (b_act, b_pressure)
    assert not touched & {"ref-r1", "ev1", "as1"}  # never decayed or pressured
    stored = after.get_node_by_id("ref-r1")
    assert stored.signals.activation.activation == entity("ref-r1", "hecate", producer=FENCED).signals.activation.activation


def test_without_the_eligibility_view_the_projection_would_move_dynamics():
    # Mutation check: feed dynamics the unfiltered graph by giving the projection a
    # legacy role. Pressure then reaches "n-c" through it, so the isolation test above
    # is load-bearing, not vacuous.
    store = InMemorySubstrateGraphStore()
    _legacy_graph(store)
    store.upsert_edge(identity_key="leak", edge=edge("leak", ("n-a", "concept"), ("n-c", "concept"), predicate="causes"))
    base = InMemorySubstrateGraphStore()
    _legacy_graph(base)
    assert _dynamics(store)[1] != _dynamics(base)[1]


def test_cognitive_view_keeps_legacy_topology_exactly():
    store = InMemorySubstrateGraphStore()
    _legacy_graph(store)
    _add_memory_structure(store)
    state = store.snapshot()
    nodes, edges = cognitive_view(state.nodes, state.edges)
    assert set(nodes) == {"n-a", "n-b", "n-c"}
    assert set(edges) == {"e1", "e2", "e3"}


def test_attention_never_offers_a_fenced_node_or_an_assertion():
    hot = {"dynamic_pressure": 0.95, "dynamic_pressure_reason": "prediction_error_seed", "prediction_error": 0.9}
    ours = concept("referent-x", "hecate", producer=FENCED, metadata=hot)
    theirs = concept("topic-x", "hecate", metadata=hot)
    assert [s.target_text for s in substrate_pressure_signals([theirs])] == ["hecate"]
    assert substrate_pressure_signals([ours]) == []
    assert not is_cognitive_node(assertion())


# ── neighborhood: projection walkable only while its assertion is accepted ──


def _walk_store(*, state: str = "provisional", revision: int = 1, edge_revision: int = 1):
    store = InMemorySubstrateGraphStore()
    for node in (entity("quill", "quill", producer=FENCED, scope="juniper"),
                 entity("retreat", "spring retreat", producer=FENCED, scope="juniper"),
                 entity("legacy-n", "legacy", scope="juniper"),
                 assertion("as1", state=state, revision=revision)):
        store.upsert_node(identity_key=node.node_id, node=node)
    store.upsert_edge(identity_key="proj", edge=edge("proj", ("quill", "entity"), ("retreat", "entity"),
                                                     edge_role="semantic_projection", assertion_id="as1",
                                                     assertion_revision=edge_revision))
    store.upsert_edge(identity_key="leg", edge=edge("leg", ("quill", "entity"), ("legacy-n", "entity"),
                                                    predicate="associated_with"))
    store.upsert_edge(identity_key="struct", edge=edge("struct", ("as1", "assertion"), ("quill", "entity"),
                                                       predicate="assertion_subject",
                                                       edge_role="assertion_structure"))
    return store


def _boundary(store) -> set[str]:
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("quill",)))
    assert not result.degraded, result.reason
    return {e.edge_id for e in result.boundary_edges}


@pytest.mark.parametrize("state,revision,edge_revision,walks", [
    ("provisional", 1, 1, True),
    ("canonical", 2, 2, True),
    ("rejected", 2, 2, False),       # retracted: the stale projection must not walk
    ("deprecated", 2, 2, False),
    ("proposed", 1, 1, False),       # never accepted
    ("provisional", 2, 1, False),    # projector failed mid-way: revision mismatch
])
def test_a_projection_walks_only_while_its_assertion_is_accepted_at_that_revision(state, revision, edge_revision,
                                                                                  walks):
    edges = _boundary(_walk_store(state=state, revision=revision, edge_revision=edge_revision))
    assert ("proj" in edges) is walks
    assert "leg" in edges  # legacy edges walk exactly as before
    assert "struct" not in edges


def test_walkable_edge_refuses_provenance_and_missing_assertions():
    prov_edge = edge("p", ("n-r", "entity"), ("n-ev", "evidence"), predicate="observed_in", edge_role="provenance")
    assert walkable_edge(prov_edge, None) is False
    proj = edge("x", ("n-a", "entity"), ("n-b", "entity"), edge_role="semantic_projection", assertion_id="gone",
                assertion_revision=1)
    assert walkable_edge(proj, None) is False


def test_recall_region_reads_drop_structure_and_unaccepted_projections():
    """#2515 review item 1, in-memory backend: legacy edges unchanged, an accepted
    projection kept, structure and a rejected projection dropped."""
    store = _walk_store(state="rejected", revision=2, edge_revision=2)
    for node in (entity("quill", "quill", scope="juniper"),):
        store.upsert_node(identity_key="v", node=node)
    region = store.read_hotspot_region(min_salience=0.0, limit_nodes=10, limit_edges=10)
    assert {e.edge_id for e in region.edges} == {"leg"}
    accepted = _walk_store(state="provisional", revision=1, edge_revision=1)
    region = accepted.read_hotspot_region(min_salience=0.0, limit_nodes=10, limit_edges=10)
    assert {e.edge_id for e in region.edges} == {"leg", "proj"}
