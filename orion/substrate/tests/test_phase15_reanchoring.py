from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.substrate_consolidation import GraphConsolidationRequestV1
from orion.core.schemas.substrate_review_queue import GraphReviewCycleBudgetV1, GraphReviewQueueItemV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_runtime import GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.store import InMemorySubstrateGraphStore, SubstrateNeighborhoodSliceV1, SubstrateQueryResultV1
from orion.substrate.frontier_curiosity import FrontierCuriosityEvaluator


class FakeSemanticStore(InMemorySubstrateGraphStore):
    def __init__(self, *, source_kind: str = "graphdb", degraded: bool = False) -> None:
        super().__init__()
        self._source_kind = source_kind
        self._degraded = degraded

    def _wrap(self, query_kind: str, result):
        return SubstrateQueryResultV1(
            query_kind=query_kind,
            slice=result,
            source_kind=self._source_kind,
            degraded=self._degraded,
            limits={"limit_nodes": 32, "limit_edges": 64},
        )

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        return self._wrap("focal_slice", self.read_focal_slice(node_ids=node_ids, max_edges=max_edges))

    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return self._wrap("hotspot_region", self.read_hotspot_region(min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges))

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return self._wrap("contradiction_region", self.read_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges))

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return self._wrap("concept_region", self.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges))

    def query_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return self._wrap(
            "provenance_neighborhood",
            self.read_provenance_neighborhood(evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges),
        )


class EmptySemanticStore(FakeSemanticStore):
    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return SubstrateQueryResultV1(
            query_kind="hotspot_region",
            source_kind="fallback",
            degraded=True,
            slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
            limits={"limit_nodes": limit_nodes, "limit_edges": limit_edges},
        )

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return SubstrateQueryResultV1(
            query_kind="concept_region",
            source_kind="fallback",
            degraded=True,
            slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
            limits={"limit_nodes": limit_nodes, "limit_edges": limit_edges},
        )

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return SubstrateQueryResultV1(
            query_kind="contradiction_region",
            source_kind="fallback",
            degraded=True,
            slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
            limits={"limit_nodes": limit_nodes, "limit_edges": limit_edges},
        )


def _seed_store(store: InMemorySubstrateGraphStore) -> None:
    now = datetime.now(timezone.utc)
    temporal = SubstrateTemporalWindowV1(observed_at=now)
    provenance = SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="pytest",
        producer="unit",
        evidence_refs=["orion"],
    )
    concept = ConceptNodeV1(
        node_id="node-concept",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.75),
        label="Core Concept",
        definition="concept",
        metadata={"concept_id": "c1", "dynamic_pressure": 0.7},
    )
    contradiction = ContradictionNodeV1(
        node_id="node-contradiction",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.85),
        summary="conflict",
        involved_node_ids=["node-concept", "node-concept"],
        metadata={"dynamic_pressure": 0.8},
    )
    edge = SubstrateEdgeV1(
        edge_id="edge-rel",
        source=NodeRefV1(node_id=concept.node_id, node_kind="concept"),
        target=NodeRefV1(node_id=contradiction.node_id, node_kind="contradiction"),
        predicate="contradicts",
        temporal=temporal,
        confidence=0.9,
        salience=0.7,
        provenance=provenance,
    )
    store.upsert_node(identity_key="concept|orion|orion|c1", node=concept)
    store.upsert_node(identity_key="contradiction|orion|orion|x", node=contradiction)
    store.upsert_edge(identity_key=f"{concept.node_id}|contradicts|{contradiction.node_id}", edge=edge)


def test_consolidation_reports_semantic_source_and_explicit_fallback() -> None:
    graphdb_store = FakeSemanticStore(source_kind="graphdb", degraded=False)
    _seed_store(graphdb_store)
    evaluator = GraphConsolidationEvaluator(store=graphdb_store)
    request = GraphConsolidationRequestV1(
        anchor_scope="orion",
        subject_ref="orion",
        target_zone="concept_graph",
        reason_for_review="phase15-test",
    )
    execution = evaluator.consolidate(request=request)
    assert execution.semantic_source == "graphdb"
    assert any(note.startswith("semantic_source:") for note in execution.result.notes)

    fallback_store = EmptySemanticStore(source_kind="fallback", degraded=True)
    _seed_store(fallback_store)
    fallback_eval = GraphConsolidationEvaluator(store=fallback_store)
    fallback_exec = fallback_eval.consolidate(request=request)
    assert fallback_exec.semantic_source == "local_fallback"
    assert fallback_exec.semantic_degraded is True


def test_curiosity_region_selection_uses_semantic_store_and_keeps_bounded_behavior() -> None:
    store = FakeSemanticStore(source_kind="graphdb", degraded=False)
    _seed_store(store)
    evaluator = FrontierCuriosityEvaluator(store=store, max_focal_nodes=4, max_focal_edges=4)
    result = evaluator.evaluate(
        anchor_scope="orion",
        subject_ref="orion",
        operator_requested=True,
    )
    assert any(note.startswith("semantic_source:") for note in result.notes)
    assert len(result.signals) >= 1
    for signal in result.signals:
        assert len(signal.focal_node_refs) <= 4
        assert len(signal.focal_edge_refs) <= 4


def test_runtime_review_audit_exposes_semantic_source_without_changing_queue_controls() -> None:
    store = FakeSemanticStore(source_kind="graphdb", degraded=False)
    _seed_store(store)
    queue = GraphReviewQueue(max_items=20)
    now = datetime.now(timezone.utc)
    queue_item = GraphReviewQueueItemV1(
        focal_node_refs=["node-concept"],
        focal_edge_refs=["edge-rel"],
        anchor_scope="orion",
        subject_ref="orion",
        target_zone="concept_graph",
        originating_decision_id="d1",
        originating_request_id="r1",
        reason_for_revisit="test",
        priority=80,
        next_review_at=now - timedelta(seconds=1),
        cycle_budget=GraphReviewCycleBudgetV1(cycle_count=0, max_cycles=3, remaining_cycles=3, no_change_cycles=0, suppress_after_low_value_cycles=2),
    )
    queue.upsert(queue_item)

    scheduler = GraphReviewScheduler(queue=queue)
    runtime = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=store),
        scheduler=scheduler,
    )

    result = runtime.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review"), now=now)
    assert result.outcome == "executed"
    assert result.audit_summary.get("semantic_source") == "graphdb"
    assert "before" in result.cycle_budget_summary and "after" in result.cycle_budget_summary
