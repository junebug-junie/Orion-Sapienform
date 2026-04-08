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
from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyRollbackRequestV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.core.schemas.substrate_review_queue import GraphReviewCycleBudgetV1, GraphReviewQueueItemV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.policy_profiles import SubstratePolicyProfileStore
from orion.substrate.query_planning import SubstrateQueryPlanner, SubstrateSemanticReadCoordinator
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_runtime import GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.store import InMemorySubstrateGraphStore


class CountingStore(InMemorySubstrateGraphStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[str, int] = {}

    def _bump(self, key: str) -> None:
        self.calls[key] = self.calls.get(key, 0) + 1

    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64):
        self._bump("hotspot_region")
        return super().query_hotspot_region(min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):
        self._bump("concept_region")
        return super().query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):
        self._bump("contradiction_region")
        return super().query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)


def _seed(store: InMemorySubstrateGraphStore) -> None:
    now = datetime.now(timezone.utc)
    temporal = SubstrateTemporalWindowV1(observed_at=now)
    provenance = SubstrateProvenanceV1(authority="local_inferred", source_kind="test", source_channel="pytest", producer="unit", evidence_refs=["p18"])
    concept = ConceptNodeV1(
        node_id="node-c",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.8),
        label="c",
        definition="c",
        metadata={"concept_id": "c1", "dynamic_pressure": 0.8, "frontier_hypothesis_marker": True},
    )
    contradiction = ContradictionNodeV1(
        node_id="node-x",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.85, salience=0.9),
        summary="x",
        involved_node_ids=["node-c", "node-c"],
        metadata={"dynamic_pressure": 0.9},
    )
    edge = SubstrateEdgeV1(
        edge_id="edge-c-x",
        source=NodeRefV1(node_id="node-c", node_kind="concept"),
        target=NodeRefV1(node_id="node-x", node_kind="contradiction"),
        predicate="contradicts",
        temporal=temporal,
        confidence=0.9,
        salience=0.8,
        provenance=provenance,
    )
    store.upsert_node(identity_key="concept|orion|orion|c1", node=concept)
    store.upsert_node(identity_key="contradiction|orion|orion|x", node=contradiction)
    store.upsert_edge(identity_key="node-c|contradicts|node-x", edge=edge)


def test_durable_policy_store_survives_restart_and_preserves_states(tmp_path) -> None:
    path = tmp_path / "policy_store.json"
    first = SubstratePolicyProfileStore(persistence_path=str(path))
    adopted = first.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=900, query_cache_enabled=False),
            activate_now=True,
            operator_id="op-1",
            rationale="persist me",
        )
    )
    first.rollback(
        SubstratePolicyRollbackRequestV1(
            rollback_target="baseline",
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            operator_id="op-1",
            rationale="rollback baseline",
        )
    )

    second = SubstratePolicyProfileStore(persistence_path=str(path))
    inspect = second.inspect(audit_limit=20)
    assert any(p.profile_id == adopted.profile_id for p in inspect.rolled_back_profiles)
    assert len(inspect.recent_audit_events) >= 3
    baseline = second.resolve(invocation_surface="operator_review", target_zone="concept_graph", operator_mode=True)
    assert baseline.mode == "baseline"


def test_query_cache_enabled_controls_coordinator_reuse() -> None:
    store = CountingStore()
    _seed(store)
    plan = SubstrateQueryPlanner.curiosity_seed(max_nodes=16, max_edges=16)

    cached = SubstrateSemanticReadCoordinator(store=store, cache_enabled=True)
    first = cached.execute(plan)
    second = cached.execute(plan)
    assert first.meta.reused_cache is False
    assert second.meta.reused_cache is True

    uncached = SubstrateSemanticReadCoordinator(store=store, cache_enabled=False)
    third = uncached.execute(plan)
    fourth = uncached.execute(plan)
    assert third.meta.reused_cache is False
    assert fourth.meta.reused_cache is False


def test_runtime_policy_wires_query_cache_flag_and_limits(tmp_path) -> None:
    store = CountingStore()
    _seed(store)
    policy_path = tmp_path / "runtime_policy.json"
    policy_store = SubstratePolicyProfileStore(persistence_path=str(policy_path))
    policy_store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(query_cache_enabled=False, query_limit_nodes=10, query_limit_edges=20, frontier_followup_allowed=False),
            activate_now=True,
            operator_id="op-2",
            rationale="runtime knobs",
        )
    )

    queue = GraphReviewQueue(max_items=20)
    now = datetime.now(timezone.utc)
    queue.upsert(
        GraphReviewQueueItemV1(
            focal_node_refs=["node-c"],
            focal_edge_refs=["edge-c-x"],
            anchor_scope="orion",
            subject_ref="orion",
            target_zone="concept_graph",
            originating_decision_id="d1",
            originating_request_id="r1",
            reason_for_revisit="phase18",
            priority=80,
            next_review_at=now - timedelta(seconds=1),
            cycle_budget=GraphReviewCycleBudgetV1(cycle_count=0, max_cycles=3, remaining_cycles=3, no_change_cycles=0, suppress_after_low_value_cycles=2),
        )
    )

    scheduler = GraphReviewScheduler(queue=queue, policy_profiles=policy_store)
    runtime = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=store),
        scheduler=scheduler,
        policy_profiles=policy_store,
    )

    result = runtime.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="operator_review", execute_frontier_followup_allowed=True), now=now)
    assert result.outcome == "executed"
    assert result.audit_summary.get("policy_query_cache_enabled") is False
    assert result.audit_summary.get("policy_mode") == "adopted"
