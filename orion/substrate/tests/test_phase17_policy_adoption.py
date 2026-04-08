from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_runtime import FrontierFollowupExecutor, GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.store import InMemorySubstrateGraphStore


class _Followup(FrontierFollowupExecutor):
    def __init__(self) -> None:
        self.invocations = 0

    def invoke_for_review(self, *, queue_item, consolidation):
        self.invocations += 1
        return True


def _seed_store(store: InMemorySubstrateGraphStore) -> None:
    now = datetime.now(timezone.utc)
    temporal = SubstrateTemporalWindowV1(observed_at=now)
    provenance = SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="pytest",
        producer="unit",
        evidence_refs=["p17"],
    )
    concept = ConceptNodeV1(
        node_id="node-concept",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.8),
        label="Core Concept",
        definition="concept",
        metadata={"concept_id": "c1", "dynamic_pressure": 0.8, "frontier_hypothesis_marker": True},
    )
    contradiction = ContradictionNodeV1(
        node_id="node-contradiction",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.85, salience=0.9),
        summary="conflict",
        involved_node_ids=["node-concept", "node-concept"],
        metadata={"dynamic_pressure": 0.9},
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


def _queue_item(*, zone: str = "concept_graph") -> GraphReviewQueueItemV1:
    now = datetime.now(timezone.utc)
    return GraphReviewQueueItemV1(
        focal_node_refs=["node-concept"],
        focal_edge_refs=["edge-rel"],
        anchor_scope="orion",
        subject_ref="orion",
        target_zone=zone,
        originating_decision_id="d1",
        originating_request_id="r1",
        reason_for_revisit="phase17",
        priority=85,
        next_review_at=now - timedelta(seconds=1),
        cycle_budget=GraphReviewCycleBudgetV1(cycle_count=0, max_cycles=3, remaining_cycles=3, no_change_cycles=0, suppress_after_low_value_cycles=2),
    )


def test_profile_validation_stage_and_activate_are_manual() -> None:
    with pytest.raises(Exception):
        SubstratePolicyOverridesV1.model_validate({"invalid_key": 1})

    store = SubstratePolicyProfileStore()
    adoption = store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=900, frontier_followup_allowed=False),
            activate_now=False,
            operator_id="op-1",
            rationale="stage first",
        )
    )
    assert adoption.action_taken == "staged"
    inspection = store.inspect()
    assert len(inspection.staged_profiles) == 1
    assert len(inspection.active_profiles) == 0

    activated = store.activate(profile_id=adoption.profile_id or "", operator_id="op-1", rationale="manual activate")
    assert activated.action_taken == "activated"
    resolution = store.resolve(invocation_surface="operator_review", target_zone="concept_graph", operator_mode=True)
    assert resolution.mode == "adopted"
    assert resolution.overrides.get("normal_revisit_seconds") == 900


def test_rollout_scope_baseline_elsewhere_and_rollback_paths() -> None:
    store = SubstratePolicyProfileStore()
    first = store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=1200),
            activate_now=True,
            operator_id="op-1",
            rationale="first",
        )
    )
    second = store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=1500),
            activate_now=True,
            operator_id="op-1",
            rationale="second",
        )
    )
    assert first.profile_id != second.profile_id

    baseline_resolution = store.resolve(invocation_surface="chat_reflective_lane", target_zone="concept_graph", operator_mode=False)
    assert baseline_resolution.mode == "baseline"

    rollback_prev = store.rollback(
        SubstratePolicyRollbackRequestV1(
            rollback_target="previous",
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            operator_id="op-1",
            rationale="revert one step",
        )
    )
    assert rollback_prev.action_taken == "rolled_back"
    assert rollback_prev.active_profile_id == first.profile_id

    rollback_base = store.rollback(
        SubstratePolicyRollbackRequestV1(
            rollback_target="baseline",
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            operator_id="op-1",
            rationale="restore baseline",
        )
    )
    assert rollback_base.action_taken == "rolled_back"
    assert rollback_base.active_profile_id is None


def test_runtime_resolution_applies_narrowly_and_keeps_strict_zone_guardrail() -> None:
    semantic_store = InMemorySubstrateGraphStore()
    _seed_store(semantic_store)
    profile_store = SubstratePolicyProfileStore()
    profile_store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(frontier_followup_allowed=False, query_limit_nodes=12, query_limit_edges=24),
            activate_now=True,
            operator_id="op-1",
            rationale="safe canary",
        )
    )

    queue = GraphReviewQueue(max_items=20)
    queue.upsert(_queue_item(zone="concept_graph"))
    scheduler = GraphReviewScheduler(queue=queue, policy_profiles=profile_store)
    followup = _Followup()
    runtime = GraphReviewRuntimeExecutor(
        queue=queue,
        consolidation_evaluator=GraphConsolidationEvaluator(store=semantic_store),
        scheduler=scheduler,
        frontier_followup_executor=followup,
        policy_profiles=profile_store,
    )
    result = runtime.execute_once(
        request=GraphReviewRuntimeRequestV1(
            invocation_surface="operator_review",
            execute_frontier_followup_allowed=True,
        )
    )
    assert result.outcome == "executed"
    assert result.audit_summary.get("policy_mode") == "adopted"
    assert followup.invocations == 0

    # Strict-zone protections remain intact on non-operator surface.
    queue.upsert(_queue_item(zone="self_relationship_graph"))
    strict_result = runtime.execute_once(request=GraphReviewRuntimeRequestV1(invocation_surface="chat_reflective_lane"))
    assert strict_result.outcome in {"operator_only", "noop"}


def test_inspection_and_comparison_hook_are_operator_readable() -> None:
    store = SubstratePolicyProfileStore()
    adopted = store.adopt(
        SubstratePolicyAdoptionRequestV1(
            rollout_scope=SubstratePolicyRolloutScopeV1(invocation_surfaces=["operator_review"], target_zones=["concept_graph"]),
            policy_overrides=SubstratePolicyOverridesV1(normal_revisit_seconds=1000),
            activate_now=True,
            operator_id="op-2",
            rationale="compare against baseline",
        )
    )
    resolution = store.resolve(invocation_surface="operator_review", target_zone="concept_graph", operator_mode=True)
    comparison = store.compare_against_baseline(resolution=resolution, baseline_summary={"avg_runtime_ms": 120.0})
    inspection = store.inspect(audit_limit=10)

    assert adopted.profile_id == comparison.active_profile_id
    assert any(note.startswith("override:") for note in comparison.comparison_notes)
    assert len(inspection.recent_audit_events) >= 2
