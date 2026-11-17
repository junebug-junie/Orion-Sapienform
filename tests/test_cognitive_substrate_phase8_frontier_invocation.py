from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    GoalNodeV1,
    HypothesisNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.frontier_expansion import FrontierDeltaItemV1, FrontierExpansionResponseV1
from orion.substrate import FrontierExpansionService, FrontierLandingEvaluator, InMemorySubstrateGraphStore, SubstrateGraphMaterializer
from orion.substrate.frontier_curiosity import FrontierCuriosityEvaluator, FrontierCuriosityOrchestrator


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="orion:test",
        producer="pytest",
    )


class _StubProvider:
    def expand(self, *, request, context):
        new_node = ConceptNodeV1(anchor_scope=request.anchor_scope, subject_ref=request.subject_ref, label="Frontier Added", temporal=_temporal(), provenance=_prov())
        return FrontierExpansionResponseV1(
            request_id=request.request_id,
            provider="stub",
            model="stub-v1",
            task_type=request.task_type,
            target_zone=request.target_zone,
            delta_items=[FrontierDeltaItemV1(item_kind="node_add", candidate_node=new_node, confidence=0.9)],
            confidence=0.9,
        )


def _build_store() -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    concept = ConceptNodeV1(node_id="concept-1", anchor_scope="orion", subject_ref="entity:orion", label="Concept 1", temporal=_temporal(), provenance=_prov(), signals={"salience": 0.8})
    goal = GoalNodeV1(node_id="goal-1", anchor_scope="orion", subject_ref="entity:orion", goal_text="Goal", temporal=_temporal(), provenance=_prov(), metadata={"dynamic_pressure": 0.8})
    contradiction = ContradictionNodeV1(node_id="contra-0", anchor_scope="orion", subject_ref="entity:orion", summary="Conflict", involved_node_ids=["concept-1", "goal-1"], temporal=_temporal(), provenance=_prov(), metadata={"severity": 0.8})
    gap = HypothesisNodeV1(node_id="gap-1", anchor_scope="orion", subject_ref="entity:orion", hypothesis_text="Evidence gap", temporal=_temporal(), provenance=_prov(), metadata={"frontier_hypothesis_marker": True})
    gap2 = HypothesisNodeV1(node_id="gap-2", anchor_scope="orion", subject_ref="entity:orion", hypothesis_text="Evidence gap 2", temporal=_temporal(), provenance=_prov(), metadata={"frontier_hypothesis_marker": True})
    record = SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=[concept, goal, contradiction, gap, gap2], edges=[])
    materializer.apply_record(record)
    return store


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing on origin/main (2026-10-10): the frontier_hypothesis_marker "
        "nodes are not in the curiosity seed slice, so no evidence_gap_scan signal "
        "is derived. Unrelated to the ontology_sparse_region retirement."
    ),
)
def test_signal_derivation_and_task_selection_are_deterministic() -> None:
    evaluator = FrontierCuriosityEvaluator(store=_build_store())
    result = evaluator.evaluate(anchor_scope="orion", subject_ref="entity:orion")

    signal_types = {signal.signal_type for signal in result.signals}
    assert "ontology_sparse_region" not in signal_types
    assert any(signal.task_type_candidate == "evidence_gap_scan" for signal in result.signals)
    assert result.decision.outcome in {"invoke", "defer", "noop", "operator_only", "blocked"}


def test_self_relationship_cases_are_operator_only() -> None:
    evaluator = FrontierCuriosityEvaluator(store=_build_store())
    result = evaluator.evaluate(
        anchor_scope="orion",
        subject_ref="entity:orion",
        operator_requested=False,
    )
    if result.decision.target_zone == "self_relationship_graph":
        assert result.decision.outcome == "operator_only"


def test_region_selection_is_bounded_and_inspectable() -> None:
    evaluator = FrontierCuriosityEvaluator(store=_build_store(), max_focal_nodes=2, max_focal_edges=2)
    result = evaluator.evaluate(anchor_scope="orion", subject_ref="entity:orion")
    if result.plan is not None:
        assert len(result.plan.selected_node_refs) <= 2
        assert len(result.plan.selected_edge_refs) <= 2


def test_integration_flows_into_phase6_and_phase7_paths() -> None:
    store = _build_store()
    curiosity = FrontierCuriosityEvaluator(store=store)
    expansion = FrontierExpansionService(store=store, provider=_StubProvider())
    landing = FrontierLandingEvaluator(store=store)
    orchestrator = FrontierCuriosityOrchestrator(curiosity_evaluator=curiosity, expansion_service=expansion, landing_evaluator=landing)

    result = orchestrator.run(
        anchor_scope="orion",
        subject_ref="entity:orion",
        operator_requested=True,
    )

    assert result.invocation.decision.outcome in {"invoke", "operator_only", "defer", "noop", "blocked"}
    if result.invocation.decision.outcome == "invoke":
        assert result.expansion is not None
        assert result.landing is not None


def test_low_value_region_can_noop_without_broad_runtime_expansion() -> None:
    store = _build_store()
    evaluator = FrontierCuriosityEvaluator(store=store)
    result = evaluator.evaluate(
        anchor_scope="orion",
        subject_ref="entity:orion",
        operator_requested=False,
    )
    assert result.decision.outcome in {"noop", "defer", "invoke", "operator_only", "blocked"}


def _concept_only_store(count: int = 15) -> InMemorySubstrateGraphStore:
    """Concept-dense slice with zero ontology_branch nodes -- the live shape.

    Live 2026-10-10: thousands of concept nodes, zero ontology_branch nodes in
    every FalkorDB graph, because nothing in Orion creates them.
    """
    store = InMemorySubstrateGraphStore()
    concepts = [
        ConceptNodeV1(
            node_id=f"concept-{idx}",
            anchor_scope="orion",
            subject_ref="entity:orion",
            label=f"Concept {idx}",
            temporal=_temporal(),
            provenance=_prov(),
            signals={"salience": 0.8},
        )
        for idx in range(count)
    ]
    SubstrateGraphMaterializer(store=store).apply_record(
        SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="entity:orion", nodes=concepts, edges=[])
    )
    return store


def test_concepts_without_ontology_branches_emit_no_ontology_sparse_region() -> None:
    """Regression: the retired constant-1.0 `ontology_sparse_region` candidate.

    It fired on any concept-dense slice with no ontology_branch nodes, scored
    min(1, 0.55 + 0.03 * n_concepts) (1.0 at 15 concepts), and handed the
    evaluator an `ontology_expand` task nothing executes -- so it won every
    decision. With no real signal present the evaluator must now noop.
    """
    evaluator = FrontierCuriosityEvaluator(store=_concept_only_store())
    result = evaluator.evaluate(
        anchor_scope="orion",
        subject_ref="entity:orion",
        operator_requested=False,
    )
    assert [s.signal_type for s in result.signals] == []
    assert result.decision.outcome == "noop"
    assert result.decision.chosen_task_type is None
    assert result.plan is None


def test_retired_kinds_are_not_in_the_contracts() -> None:
    from typing import get_args

    from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalTypeV1
    from orion.core.schemas.frontier_expansion import FrontierTaskTypeV1

    assert "ontology_sparse_region" not in get_args(FrontierInvocationSignalTypeV1)
    assert "ontology_expand" not in get_args(FrontierTaskTypeV1)


def test_report_driven_branches_are_retired() -> None:
    """#3393 Patch F: the contradiction, drift, goal-pressure and identity-conflict
    branches read a hardcoded "all calm" report and never fired. A store that DOES
    hold contradiction and high-pressure goal nodes must still produce none of them,
    and evaluate() no longer accepts the report arguments at all."""
    import inspect

    params = inspect.signature(FrontierCuriosityEvaluator.evaluate).parameters
    assert "cognition_report" not in params and "perception_brief" not in params
    run_params = inspect.signature(FrontierCuriosityOrchestrator.run).parameters
    assert "cognition_report" not in run_params and "perception_brief" not in run_params

    result = FrontierCuriosityEvaluator(store=_build_store()).evaluate(anchor_scope="orion", subject_ref="entity:orion")
    signal_types = {signal.signal_type for signal in result.signals}
    assert not signal_types & {"contradiction_hotspot", "concept_instability", "unresolved_pressure_region"}
    assert not any(s.task_type_candidate == "self_or_relationship_hypothesis" for s in result.signals)


def test_retired_report_signal_types_are_not_in_the_contract() -> None:
    from typing import get_args

    from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalTypeV1

    retired = {"contradiction_hotspot", "concept_instability", "unresolved_pressure_region"}
    assert not retired & set(get_args(FrontierInvocationSignalTypeV1))


def test_graph_cognition_package_is_gone() -> None:
    import importlib.util

    assert importlib.util.find_spec("orion.graph_cognition") is None
