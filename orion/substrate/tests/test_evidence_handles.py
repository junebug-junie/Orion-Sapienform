"""read_evidence_handles: one reference rule, three backends (PR E).

Real FalkorDB parity lives in test_neighborhood_falkor_live.py.
"""
from datetime import datetime, timedelta

import pytest
import rdflib
from pydantic import ValidationError

from orion.core.schemas.cognitive_substrate import (
    EntityNodeV1, EvidenceNodeV1, NodeRefV1, SubstrateEdgeV1, SubstrateTemporalWindowV1,
)
from orion.substrate.evals.neighborhood_fixture import NOW, PROVENANCE, TEMPORAL, concept, graph
from orion.substrate.evidence_handles import EvidenceHandleRequestV1
from orion.substrate.graphdb_store import GraphDBSubstrateStore, GraphDBSubstrateStoreConfig
from orion.substrate.routed_store import RoutedSubstrateGraphStore


def evidence(node_id, evidence_type="episode_memory"):
    return EvidenceNodeV1(node_id=node_id, anchor_scope="juniper", evidence_type=evidence_type,
        content_ref=f"{evidence_type}:{node_id}", temporal=TEMPORAL, provenance=PROVENANCE)


def link(edge_id, source, source_kind, target, target_kind, predicate, *, at=None, valid_from=None, valid_to=None):
    return SubstrateEdgeV1(edge_id=edge_id, predicate=predicate,
        source=NodeRefV1(node_id=source, node_kind=source_kind),
        target=NodeRefV1(node_id=target, node_kind=target_kind),
        temporal=SubstrateTemporalWindowV1(observed_at=at or NOW, valid_from=valid_from, valid_to=valid_to),
        provenance=PROVENANCE)


def day(n):
    return NOW + timedelta(days=n)


def fixture():
    hecate = EntityNodeV1(node_id="hecate", label="hecate", anchor_scope="orion", entity_type="machine",
        promotion_state="provisional", temporal=TEMPORAL, provenance=PROVENANCE)
    nodes = [hecate, concept("gpu"), evidence("ev-m1"), evidence("ev-m2"), evidence("ev-m3"),
             evidence("ev-r1", "reverie"), evidence("old"), evidence("future"), evidence("ev-t1", "topic_foundry_run_topic")]
    edges = [
        link("e-m1", "hecate", "entity", "ev-m1", "evidence", "observed_in", valid_from=day(-3)),
        link("e-m2", "hecate", "entity", "ev-m2", "evidence", "observed_in", valid_from=day(-1)),
        # No valid_from: ordered by observed_at instead.
        link("e-m3", "hecate", "entity", "ev-m3", "evidence", "observed_in", at=day(-2)),
        link("e-r1", "hecate", "entity", "ev-r1", "evidence", "observed_in", valid_from=day(-1)),
        link("e-old", "hecate", "entity", "old", "evidence", "observed_in", valid_from=day(-9), valid_to=day(-5)),
        link("e-future", "hecate", "entity", "future", "evidence", "observed_in", valid_from=day(3)),
        # Wrong direction for each predicate: not a provenance shape.
        link("e-backwards", "ev-m1", "evidence", "hecate", "entity", "observed_in"),
        link("e-supports-out", "hecate", "entity", "ev-t1", "evidence", "supports"),
        # Evidence -supports-> concept is the shape topic-foundry writes today.
        link("e-t1", "ev-t1", "evidence", "gpu", "concept", "supports", valid_from=day(-4)),
        # Semantic edge: never a handle.
        link("e-sem", "hecate", "entity", "gpu", "concept", "associated_with"),
    ]
    return graph(nodes, edges)


def handles(result):
    return [(h.node_id, h.edge_id, h.evidence_type) for h in result.handles]


def test_rule_validity_order_and_shapes():
    result = fixture().read_evidence_handles(EvidenceHandleRequestV1(node_ids=("hecate", "gpu"), at=NOW))
    assert handles(result) == [
        ("gpu", "e-t1", "topic_foundry_run_topic"),
        ("hecate", "e-m2", "episode_memory"), ("hecate", "e-r1", "reverie"),
        ("hecate", "e-m3", "episode_memory"), ("hecate", "e-m1", "episode_memory"),
    ]
    assert result.complete_for_request and not result.truncated and not result.degraded
    first = result.handles[1]
    assert first.content_ref == "episode_memory:ev-m2" and first.predicate == "observed_in"
    assert first.evidence_node_id == "ev-m2"


def test_as_of_reads_the_past_and_future_windows():
    store = fixture()
    then = store.read_evidence_handles(EvidenceHandleRequestV1(node_ids=("hecate",), at=day(-6)))
    assert [h.edge_id for h in then.handles] == ["e-old"]
    later = store.read_evidence_handles(EvidenceHandleRequestV1(node_ids=("hecate",), at=day(4)))
    assert "e-future" in [h.edge_id for h in later.handles] and "e-old" not in [h.edge_id for h in later.handles]


def test_type_filter_and_truncation_receipt():
    store = fixture()
    result = store.read_evidence_handles(EvidenceHandleRequestV1(
        node_ids=("hecate",), evidence_types=("episode_memory",), per_node_limit=2, at=NOW))
    assert [h.edge_id for h in result.handles] == ["e-m2", "e-m3"]
    assert result.truncated and result.truncated_node_ids == ("hecate",)
    assert not result.complete_for_request and result.reason == "budget_exhausted"
    exact = store.read_evidence_handles(EvidenceHandleRequestV1(
        node_ids=("hecate",), evidence_types=("reverie",), per_node_limit=1, at=NOW))
    assert [h.edge_id for h in exact.handles] == ["e-r1"] and exact.complete_for_request


def test_missing_or_non_semantic_nodes_degrade_instead_of_empty_success():
    result = fixture().read_evidence_handles(EvidenceHandleRequestV1(node_ids=("absent", "ev-m1", "gpu"), at=NOW))
    assert result.degraded and not result.complete_for_request
    assert result.missing_node_ids == ("absent", "ev-m1")
    assert [h.node_id for h in result.handles] == ["gpu"]


@pytest.mark.parametrize("kwargs", [
    {"node_ids": ()}, {"node_ids": tuple(f"n{i}" for i in range(17))},
    {"per_node_limit": 0}, {"per_node_limit": 17}, {"evidence_types": ()},
    {"at": datetime(2026, 10, 6)},
])
def test_request_bounds(kwargs):
    with pytest.raises(ValidationError):
        EvidenceHandleRequestV1(**({"node_ids": ("a",)} | kwargs))


def test_routed_reads_primary_only():
    store = fixture()

    class Shadow:
        def read_evidence_handles(self, request):
            pytest.fail("shadow read forbidden")

    routed = RoutedSubstrateGraphStore(primary=store, shadow=Shadow())
    assert routed.read_evidence_handles(EvidenceHandleRequestV1(node_ids=("gpu",), at=NOW)).handles


def sparql_store(memory, cap=None):
    dataset = rdflib.Dataset()
    store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://unused"))
    store._update = dataset.update
    for node in memory._nodes.values():
        store.upsert_node(identity_key=node.node_id, node=node)
    for item in memory._edges.values():
        store.upsert_edge(identity_key=item.edge_id, edge=item)

    def select(query):
        rows = [{str(k): {"value": str(v)} for k, v in row.asdict().items()} for row in dataset.query(query)]
        return rows if cap is None else rows[:cap]

    store._select = select
    store._cache._nodes.clear()
    store._cache._edges.clear()
    store.snapshot = lambda: pytest.fail("full hydration forbidden")
    return store


@pytest.mark.parametrize("request_kwargs", [
    {"node_ids": ("hecate", "gpu")},
    {"node_ids": ("hecate",), "per_node_limit": 2, "evidence_types": ("episode_memory",)},
    {"node_ids": ("hecate", "absent"), "at": day(-6)},
])
def test_sparql_parity_with_reference(request_kwargs):
    memory = fixture()
    request = EvidenceHandleRequestV1(**({"at": NOW} | request_kwargs))
    expected = memory.read_evidence_handles(request)
    actual = sparql_store(memory).read_evidence_handles(request)
    assert actual.source_kind == "graphdb"
    assert handles(actual) == handles(expected)
    assert (actual.truncated_node_ids, actual.missing_node_ids, actual.degraded) == (
        expected.truncated_node_ids, expected.missing_node_ids, expected.degraded)


def test_backend_failure_is_degraded_without_secret_text():
    store = sparql_store(fixture())

    def fail(query):
        raise RuntimeError("password=hunter2")

    store._select = fail
    result = store.read_evidence_handles(EvidenceHandleRequestV1(node_ids=("hecate",)))
    assert result.degraded and not result.handles and "hunter2" not in (result.reason or "")
